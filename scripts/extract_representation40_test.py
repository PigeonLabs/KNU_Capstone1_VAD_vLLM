"""Identical test crops across frozen/selected visual towers, after normal freeze."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from ipad_vad.data import frames_in_order
from ipad_vad.learned_visual import load_mobile_visual,configure_adaptation
from ipad_vad.representation_data import ViewDataset,mobile_transform
from prepare_representation40 import ART,OUT,sha,write
from experiment40_normal import RUNS


def verify_normal():
    for name in ['pre_normal_models_protocol','normal_models_checkpoint']:
        for p,h in json.loads((OUT/f'{name}.json').read_text())['file_sha256'].items():assert sha(p)==h,p


def prepare():
    verify_normal();cfg=json.loads(Path('configs/experiment40_representation.json').read_text());views=[];sources={};sequences=[]
    for scene in cfg['scenes']:
        for seqdir in sorted((Path(cfg['data_root'])/scene/'testing/frames').iterdir()):
            if not seqdir.is_dir():continue
            seq=seqdir.name;source=Path(cfg['source_features'][scene])/f'testing_{seq}.npz';sources[str(source)]=sha(source)
            with np.load(source,allow_pickle=False) as f:d=dict(f)
            images=frames_in_order(seqdir);assert len(images)==int(d['frame_count']);counts={}
            for kind in ['global','crop']:
                counts[kind]=len(d['indices']) if kind=='global' else len(d['boxes'])
                for j in range(counts[kind]):
                    step=j if kind=='global' else int(d['object_frames'][j]);frame=int(d['indices'][step]);box=None if kind=='global' else d['boxes'][j].tolist()
                    if box is not None:assert np.isfinite(box).all() and 0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1
                    views.append({'scene':scene,'sequence':seq,'partition':'testing','kind':kind,'sample_index':step,'source_frame':frame,'detection_index':None if kind=='global' else j,'box':box,'path':str(images[frame])})
            sequences.append({'scene':scene,'sequence':seq,'source_frames':len(images),'views':counts})
    path=ART/'test_views.jsonl';text=''.join(json.dumps(v,separators=(',',':'))+'\n' for v in views)
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    record={'normal_checkpoint_sha256':sha(OUT/'normal_models_checkpoint.json'),'test_source_feature_sha256':sources,'view_manifest':str(path),'view_manifest_sha256':sha(path),'view_count':len(views),'sequences':sequences,'labels_opened':False}
    path=OUT/'test_data_manifest.json'
    if path.exists():assert json.loads(path.read_text())==record
    else:write(path,record)
    print('Prepared test views without labels',len(views),len(sequences),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',choices=RUNS);p.add_argument('--prepare',action='store_true');args=p.parse_args()
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and '/test_label/' in str(args[0]):raise RuntimeError('No label access in feature extraction')
    sys.addaudithook(guard)
    if args.prepare:prepare();return
    if args.run is None:p.error('--run or --prepare required')
    verify_normal();run=args.run;torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    cfg=json.loads(Path('configs/experiment40_representation.json').read_text());normal=json.loads((OUT/'data_manifest.json').read_text());manifest=json.loads((OUT/'test_data_manifest.json').read_text());assert sha(manifest['view_manifest'])==manifest['view_manifest_sha256'];rows=[json.loads(s) for s in Path(manifest['view_manifest']).read_text().splitlines()]
    checkpoint_sha=None
    if run=='A':
        from transformers import CLIPModel,CLIPProcessor
        record=json.loads(Path('artifacts/model_paths.json').read_text())['openai/clip-vit-base-patch32'];model=CLIPModel.from_pretrained(record['local_path'],local_files_only=True).cuda().eval();processor=CLIPProcessor.from_pretrained(record['local_path'],local_files_only=True).image_processor
        assert processor.size['shortest_edge']==224 and processor.crop_size=={'height':224,'width':224}
        transform=T.Compose([T.Resize(224,interpolation=InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize(processor.image_mean,processor.image_std)]);encode=lambda x:model.get_image_features(pixel_values=x)
    else:
        model,_=load_mobile_visual(normal['model']);transform=mobile_transform()
        if run!='B':
            training=json.loads((OUT/f'{run}_training.json').read_text());checkpoint_sha=training['checkpoint_sha256'];assert sha(training['checkpoint'])==checkpoint_sha
            scope=configure_adaptation(model,'lora' if run.startswith('C') else 'full',cfg['lora_rank'],cfg['lora_alpha']);assert scope==training['scope'];saved=torch.load(training['checkpoint'],map_location='cpu',weights_only=True);assert saved['epoch']==training['selected_epoch'] and saved['run']==run;model.load_state_dict(saved['visual'],strict=True)
        model.cuda().eval();encode=model
    model.requires_grad_(False);start=time.perf_counter();torch.cuda.reset_peak_memory_stats();completed=[]
    for row in manifest['sequences']:
        scene,seq=row['scene'],row['sequence'];target=ART/run/'features'/scene/f'testing_{seq}.npz'
        if target.exists():raise RuntimeError('Refuse overwrite: '+str(target))
        selected=[r for r in rows if r['scene']==scene and r['sequence']==seq];loader=DataLoader(ViewDataset(selected,transform),batch_size=64,shuffle=False,num_workers=0);features=[]
        with torch.inference_mode():
            for images,_ in loader:features.append(torch.nn.functional.normalize(encode(images.cuda()).float(),dim=-1).cpu().numpy())
        z=np.concatenate(features);assert np.isfinite(z).all();np.testing.assert_allclose(np.linalg.norm(z,axis=1),1,rtol=1e-5,atol=1e-6)
        source=Path(cfg['source_features'][scene])/target.name;assert sha(source)==manifest['test_source_feature_sha256'][str(source)]
        with np.load(source,allow_pickle=False) as f:data=dict(f)
        n=len(data['indices']);assert [r['kind'] for r in selected]==['global']*n+['crop']*len(data['boxes']);data['global_features']=z[:n];data['crop_features']=z[n:];target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**data);completed.append({'scene':scene,'sequence':seq,'views':len(z),'sha256':sha(target)});print(run,scene,seq,len(z),flush=True)
    torch.cuda.synchronize();write(OUT/f'{run}_test_extraction.json',{'run':run,'labels_opened':False,'normal_checkpoint_sha256':manifest['normal_checkpoint_sha256'],'checkpoint_sha256':checkpoint_sha,'sequences':completed,'views':sum(r['views'] for r in completed),'elapsed_seconds':time.perf_counter()-start,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'precision':'float32_forward_and_normalization','timing_scope':'model loaded; image IO/preprocessing/GPU/serialization; not streaming'})

if __name__=='__main__':main()
