"""Extract frozen large encoders on identical views; hash-verified shard resume."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from ipad_vad.large_visual import load_large_visual
from ipad_vad.representation_data import view_image
from candidates45_common import OUT,ART,CFG,config,guard,freeze,verify_record,sha,write


class Views(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return view_image(self.rows[i])


class Collate:
    def __init__(self,processor):self.processor=processor
    def __call__(self,images):return self.processor(images=images,return_tensors='pt')['pixel_values']


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',choices=list(config()['candidates']),required=True)
    p.add_argument('--partition',choices=['normal','test'],default='normal');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    guard(args.partition=='normal');torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    cfg=config();name=args.candidate;record=json.loads((OUT/f'{name}_model.json').read_text())
    for n,h in record['sha256'].items():assert sha(Path(record['local_path'])/n)==h
    source_manifest=Path('results/experiment40')/('data_manifest.json' if args.partition=='normal' else 'test_data_manifest.json')
    manifest=json.loads(source_manifest.read_text());assert sha(manifest['view_manifest'])==manifest['view_manifest_sha256']
    rows=[json.loads(s) for s in Path(manifest['view_manifest']).read_text().splitlines()]
    if args.partition=='test':
        verify_record(OUT/'normal_models_checkpoint.json');verify_record(OUT/'selection_checkpoint.json')
    files=[CFG,Path(__file__),Path('scripts/candidates45_common.py'),Path('src/ipad_vad/large_visual.py'),
           Path('src/ipad_vad/representation_data.py'),Path('docs/EXPERIMENT45_PLAN.md'),source_manifest,
           Path(manifest['view_manifest']),OUT/f'{name}_model.json']
    protocol=OUT/f'{name}_{args.partition}_extraction_protocol.json'
    if not args.smoke:freeze(protocol,files,candidate=name,partition=args.partition,labels_opened=False)
    model,processor,info=load_large_visual(record);model.cuda().eval().requires_grad_(False)
    meta={'candidate':name,'family':record['family'],'parameters':sum(p.numel() for p in model.parameters()),
          'processor':processor.to_dict(),'precision':'float32_forward_and_normalization',
          'missing_keys':info['missing_keys'],'unused_checkpoint_keys':len(info['unexpected_keys'])}
    if args.smoke:
        images=[view_image(r) for r in rows[:32]];pixels=Collate(processor)(images).cuda()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            z=model(pixels);torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(5):z=model(pixels)
            torch.cuda.synchronize()
        assert torch.isfinite(z).all();meta.update(input_shape=list(pixels.shape),feature_dimension=z.shape[1],
            batch32_seconds=(time.perf_counter()-start)/5,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write(OUT/f'{name}_smoke.json',meta);print(json.dumps(meta),flush=True);return
    completed=[];source_cfg=json.loads(Path('configs/experiment40_representation.json').read_text());jobs=[];pending_rows=[];batches=[]
    for item in manifest['sequences']:
        scene,seq=item['scene'],item['sequence'];prefix='training' if args.partition=='normal' else 'testing'
        target=ART/name/'features'/scene/f'{prefix}_{seq}.npz';journal=ART/name/'journals'/scene/f'{prefix}_{seq}.json'
        selected=[r for r in rows if r['scene']==scene and r['sequence']==seq]
        source=Path(source_cfg['source_features'][scene])/target.name
        hashes=manifest['normal_source_feature_sha256' if args.partition=='normal' else 'test_source_feature_sha256']
        assert sha(source)==hashes[str(source)]
        if journal.exists():
            saved=json.loads(journal.read_text());assert saved['protocol_sha256']==sha(protocol)
            assert target.exists() and sha(target)==saved['sha256'];completed.append(saved)
            print(name,scene,seq,'verified resume',flush=True);continue
        if target.exists():raise RuntimeError('Unjournaled shard, inspect rather than overwrite: '+str(target))
        start_index=len(pending_rows);pending_rows.extend(selected)
        batches.extend([list(range(i,min(i+cfg['batch_size'],len(pending_rows)))) for i in range(start_index,len(pending_rows),cfg['batch_size'])])
        jobs.append((scene,seq,target,journal,selected,source))
    loader=DataLoader(Views(pending_rows),batch_sampler=batches,num_workers=cfg['workers'],
        collate_fn=Collate(processor),pin_memory=True,multiprocessing_context='spawn' if cfg['workers'] else None)
    iterator=iter(loader)
    for scene,seq,target,journal,selected,source in jobs:
        features=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for _ in range((len(selected)+cfg['batch_size']-1)//cfg['batch_size']):
                pixels=next(iterator)
                z=torch.nn.functional.normalize(model(pixels.cuda(non_blocking=True)).float(),dim=-1)
                features.append(z.cpu().numpy())
        z=np.concatenate(features);assert np.isfinite(z).all()
        np.testing.assert_allclose(np.linalg.norm(z,axis=1),1,rtol=1e-5,atol=1e-6)
        with np.load(source,allow_pickle=False) as archive:data=dict(archive)
        n=len(data['indices']);assert [r['kind'] for r in selected]==['global']*n+['crop']*len(data['boxes'])
        data['global_features']=z[:n];data['crop_features']=z[n:];target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix('.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,**data)
        temp.replace(target);torch.cuda.synchronize()
        saved={'scene':scene,'sequence':seq,'views':len(z),'dimension':z.shape[1],'sha256':sha(target),
            'protocol_sha256':sha(protocol),'elapsed_seconds':time.perf_counter()-start,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        write(journal,saved);completed.append(saved);print(name,scene,seq,len(z),round(saved['elapsed_seconds'],2),flush=True)
    assert next(iterator,None) is None
    completed.sort(key=lambda r:(r['scene'],r['sequence']))
    assert sum(r['views'] for r in completed)==manifest['view_count']
    write(OUT/f'{name}_{args.partition}_extraction.json',{**meta,'normal_only':args.partition=='normal','labels_opened':False,
        'sequences':completed,'views':sum(r['views'] for r in completed),
        'elapsed_seconds':sum(r['elapsed_seconds'] for r in completed),
        'gpu_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in completed),
        'timing_scope':'per-video image IO, workers, preprocessing, FP32 encoding, feature serialization; excludes checkpoint load; not streaming FPS'})


if __name__=='__main__':main()
