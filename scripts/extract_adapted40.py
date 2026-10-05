"""Extract selected normal C/D features; all process/box/track arrays stay fixed."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ipad_vad.learned_visual import load_mobile_visual,configure_adaptation
from ipad_vad.representation_data import read_views,ViewDataset,mobile_transform
from prepare_representation40 import sha,write,ART,OUT


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['C','D'],required=True);p.add_argument('--seed',type=int,required=True);args=p.parse_args();run=f'{args.arm}_s{args.seed}'
    cfg=json.loads(Path('configs/experiment40_representation.json').read_text());manifest=json.loads((OUT/'data_manifest.json').read_text());training=json.loads((OUT/f'{run}_training.json').read_text());checkpoint=Path(training['checkpoint'])
    assert sha(checkpoint)==training['checkpoint_sha256'];assert sha(manifest['view_manifest'])==manifest['view_manifest_sha256']
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    model,_=load_mobile_visual(manifest['model']);scope=configure_adaptation(model,'lora' if args.arm=='C' else 'full',cfg['lora_rank'],cfg['lora_alpha']);assert scope==training['scope']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);assert saved['run']==run and saved['epoch']==training['selected_epoch'];model.load_state_dict(saved['visual'],strict=True);model.requires_grad_(False);model.cuda().eval()
    rows=read_views(manifest['view_manifest'],['representation_train','representation_validation','normal_calibration']);start=time.perf_counter();torch.cuda.reset_peak_memory_stats();completed=[]
    for scene in cfg['scenes']:
        for seq in manifest['subsplits'][scene]['downstream_fit']+manifest['subsplits'][scene]['normal_calibration']:
            target=ART/run/'features'/scene/f'training_{seq}.npz'
            if target.exists():raise RuntimeError('Refuse to overwrite features: '+str(target))
            selected=[r for r in rows if r['scene']==scene and r['sequence']==seq];loader=DataLoader(ViewDataset(selected,mobile_transform()),batch_size=64,shuffle=False,num_workers=0);features=[]
            with torch.inference_mode():
                for images,idx in loader:features.append(torch.nn.functional.normalize(model(images.cuda()).float(),dim=-1).cpu().numpy())
            z=np.concatenate(features);source=Path(cfg['source_features'][scene])/target.name;assert sha(source)==manifest['normal_source_feature_sha256'][str(source)]
            with np.load(source,allow_pickle=False) as f:data=dict(f)
            n=len(data['indices']);assert [r['kind'] for r in selected]==['global']*n+['crop']*len(data['boxes']);data['global_features']=z[:n];data['crop_features']=z[n:]
            target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**data);completed.append({'scene':scene,'sequence':seq,'views':len(z),'sha256':sha(target)});print(run,scene,seq,len(z),flush=True)
    torch.cuda.synchronize();write(OUT/f'{run}_normal_extraction.json',{'run':run,'normal_only':True,'checkpoint_sha256':training['checkpoint_sha256'],'selected_epoch':training['selected_epoch'],'sequences':completed,'views':sum(r['views'] for r in completed),'elapsed_seconds':time.perf_counter()-start,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'precision':'float32_forward_and_normalization','timing_scope':'model already loaded; includes sequential image IO/preprocessing/GPU/feature serialization; not camera streaming'})

if __name__=='__main__':main()
