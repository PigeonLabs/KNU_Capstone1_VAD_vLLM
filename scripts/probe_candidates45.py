"""Normal-only synthetic probe and pre-test model selection, never true defect GT."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from ipad_vad.large_visual import load_large_visual,probe_image
from ipad_vad.representation_data import view_image
from ipad_vad.scoring import Subspace
from extract_candidates45 import Collate
from candidates45_common import OUT,ART,CFG,config,guard,freeze,verify_record,sha,write

VARIANTS=['clean','benign','occlusion','local_shuffle','local_noise']


def prepare():
    cfg=config();manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text())
    assert sha(manifest['view_manifest'])==manifest['view_manifest_sha256']
    rows=[json.loads(s) for s in Path(manifest['view_manifest']).read_text().splitlines()]
    rng=np.random.default_rng(cfg['seed']);chosen={'train':[],'validation':[]};counts=[]
    for split,partition,num in [('train','representation_train',cfg['probe']['train_views_per_scene_kind']),
                                ('validation','representation_validation',cfg['probe']['validation_views_per_scene_kind'])]:
        for scene in cfg['scenes']:
            for kind in ['global','crop']:
                groups={}
                for row in rows:
                    if row['partition']==partition and row['scene']==scene and row['kind']==kind:
                        groups.setdefault(row['sequence'],[]).append(row)
                order=rng.permutation(sorted(groups));selected=[]
                for i in range(num):
                    seq=order[i%len(order)];row=dict(groups[seq][int(rng.integers(len(groups[seq])))]);row['probe_seed']=int(rng.integers(0,2**31-1));selected.append(row)
                chosen[split].extend(selected);counts.append({'split':split,'scene':scene,'kind':kind,'views':num,
                    'unique_views':len({(r['sequence'],r['sample_index'],r['detection_index']) for r in selected}),
                    'videos':sorted(groups),'sampling':'video_balanced_with_replacement'})
    dest=ART/'probe_views.json';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert json.loads(dest.read_text())==chosen
    else:write(dest,chosen)
    files=[CFG,Path(__file__),Path('scripts/candidates45_common.py'),Path('src/ipad_vad/large_visual.py'),
        Path('src/ipad_vad/scoring.py'),Path('src/ipad_vad/representation_data.py'),Path('docs/EXPERIMENT45_PLAN.md'),
        Path('results/experiment40/data_manifest.json'),dest]
    freeze(OUT/'probe_protocol.json',files,normal_only=True,calibration_excluded=True,test_excluded=True,counts=counts,
        proxy_not_ground_truth=True)
    return chosen


class ProbeViews(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)*len(VARIANTS)
    def __getitem__(self,i):
        row=self.rows[i//len(VARIANTS)];variant=VARIANTS[i%len(VARIANTS)]
        return probe_image(view_image(row),variant,row['probe_seed'])


def cached(rows,name):
    result=[];loaded={}
    for r in rows:
        key=(r['scene'],r['sequence'])
        if key not in loaded:
            path=ART/name/'features'/r['scene']/f'training_{r["sequence"]}.npz'
            with np.load(path,allow_pickle=False) as f:loaded[key]={k:f[k] for k in ['global_features','crop_features']}
        arr=loaded[key]['global_features' if r['kind']=='global' else 'crop_features']
        result.append(arr[r['sample_index'] if r['kind']=='global' else r['detection_index']])
    return np.stack(result)


def run(name):
    guard();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    chosen=prepare();verify_record(OUT/'probe_protocol.json');verify_record(OUT/f'{name}_normal_extraction_protocol.json')
    extraction=json.loads((OUT/f'{name}_normal_extraction.json').read_text())
    for row in extraction['sequences']:
        assert sha(ART/name/'features'/row['scene']/f'training_{row["sequence"]}.npz')==row['sha256']
    destination=OUT/f'{name}_probe.json'
    if destination.exists():raise RuntimeError('Probe exists; verify instead of overwrite')
    model,processor,_=load_large_visual(json.loads((OUT/f'{name}_model.json').read_text()));model.cuda().eval().requires_grad_(False)
    loader=DataLoader(ProbeViews(chosen['validation']),batch_size=config()['batch_size'],num_workers=2,
        multiprocessing_context='spawn',collate_fn=Collate(processor),pin_memory=True)
    z=[];seconds=0.;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        # Separate warmup so timed batches do not depend on first CUDA kernel compilation.
        first=Collate(processor)([view_image(chosen['validation'][0])]).cuda();model(first);torch.cuda.synchronize()
        for pixels in loader:
            pixels=pixels.cuda();torch.cuda.synchronize();t=time.perf_counter()
            output=torch.nn.functional.normalize(model(pixels).float(),dim=-1);torch.cuda.synchronize()
            seconds+=time.perf_counter()-t;z.append(output.cpu().numpy())
    z=np.concatenate(z).reshape(len(chosen['validation']),len(VARIANTS),-1);assert np.isfinite(z).all()
    original=cached(chosen['validation'],name)
    np.testing.assert_allclose(z[:,0],original,rtol=2e-3,atol=3e-5) # Different batch layouts may round FP32 GEMM slightly.
    train=cached(chosen['train'],name);rows=[];arrays={'features':z,'train_features':train}
    cfg=config()['probe']
    for scene in config()['scenes']:
        for kind in ['global','crop']:
            ti=[i for i,r in enumerate(chosen['train']) if r['scene']==scene and r['kind']==kind]
            vi=[i for i,r in enumerate(chosen['validation']) if r['scene']==scene and r['kind']==kind]
            space=Subspace(train[ti],cfg['pca_variance'],cfg['pca_max_rank']);res=space.residual(z[vi].reshape(-1,z.shape[-1])).reshape(len(vi),len(VARIANTS))
            for j,corruption in enumerate(VARIANTS[2:],2):
                negative=res[:,:2].reshape(-1);positive=res[:,j]
                auc=float(roc_auc_score(np.r_[np.zeros(len(negative)),np.ones(len(positive))],np.r_[negative,positive]))
                rows.append({'scene':scene,'kind':kind,'corruption':corruption,'auroc':auc,'train_samples':len(ti),
                    'validation_samples':len(vi),'rank':space.rank,'clean_residual_mean':float(res[:,0].mean()),
                    'benign_residual_mean':float(res[:,1].mean()),'corrupt_residual_mean':float(positive.mean())})
            arrays[f'{scene}_{kind}_residuals']=res
    path=ART/name/'probe.npz';np.savez_compressed(path,**arrays)
    write(destination,{'candidate':name,'rows':rows,'selection_score':float(np.mean([r['auroc'] for r in rows])),
        'probe_protocol_sha256':sha(OUT/'probe_protocol.json'),'normal_extraction_sha256':sha(OUT/f'{name}_normal_extraction.json'),
        'cache_path':str(path),'cache_sha256':sha(path),'encoder_seconds':seconds,
        'encoded_views':len(chosen['validation'])*len(VARIANTS),'total_seconds':time.perf_counter()-start,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'normal_only':True,'test_used':False,
        'limitation':'Synthetic local appearance perturbation proxy, not real anomaly labels or semantic anomaly recall.'})
    print(name,'proxy score',np.mean([r['auroc'] for r in rows]),flush=True)


def select():
    guard();verify_record(OUT/'probe_protocol.json');records=[];files=[CFG,OUT/'probe_protocol.json',Path(__file__)]
    for name in config()['candidates']:
        path=OUT/f'{name}_probe.json';r=json.loads(path.read_text());assert r['normal_only'] and not r['test_used']
        assert r['probe_protocol_sha256']==sha(OUT/'probe_protocol.json')
        assert sha(r['cache_path'])==r['cache_sha256'];records.append(r);files.extend([path,Path(r['cache_path'])])
    best=max(r['selection_score'] for r in records);eligible=[r for r in records if r['selection_score']>=best-.005]
    selected=min(eligible,key=lambda r:(r['encoder_seconds']/r['encoded_views'],r['candidate']))
    dest=OUT/'selection.json';result={'selected':selected['candidate'],'normal_only':True,'test_used':False,
        'proxy_score':selected['selection_score'],'scores':[{k:r[k] for k in ['candidate','selection_score','encoder_seconds','encoded_views']} for r in records],
        'rule':config()['probe']['selection'],'near_ties':[r['candidate'] for r in eligible],
        'note':'Selection frozen before test extraction/evaluation; no real anomaly labels used. Synthetic proxy may not transfer.'}
    if dest.exists():assert json.loads(dest.read_text())==result
    else:write(dest,result)
    freeze(OUT/'selection_checkpoint.json',files+[dest],normal_only=True,selected=selected['candidate'])
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',choices=list(config()['candidates']));p.add_argument('--select',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=4):
        if args.select:select()
        elif args.candidate:run(args.candidate)
        else:prepare()
