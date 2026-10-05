"""Extract normal A/B visuals with identical cached boxes, without test access."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from ipad_vad.learned_visual import load_mobile_visual
from ipad_vad.representation_data import read_views,ViewDataset,mobile_transform
from prepare_representation40 import sha,write,ART,OUT


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['A','B'],required=True);args=p.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    cfg=json.loads(Path('configs/experiment40_representation.json').read_text());manifest=json.loads((OUT/'data_manifest.json').read_text())
    assert sha(manifest['view_manifest'])==manifest['view_manifest_sha256']
    rows=read_views(manifest['view_manifest'],['representation_train','representation_validation','normal_calibration'])
    if args.arm=='A':
        from transformers import CLIPModel,CLIPProcessor
        record=json.loads(Path('artifacts/model_paths.json').read_text())['openai/clip-vit-base-patch32'];model=CLIPModel.from_pretrained(record['local_path'],local_files_only=True).cuda().eval()
        processor=CLIPProcessor.from_pretrained(record['local_path'],local_files_only=True).image_processor
        assert processor.size['shortest_edge']==224 and processor.crop_size=={'height':224,'width':224}
        transform=T.Compose([T.Resize(224,interpolation=InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize(processor.image_mean,processor.image_std)])
        encode=lambda x:model.get_image_features(pixel_values=x)
    else:
        model,preprocess=load_mobile_visual(manifest['model']);model=model.cuda().eval();transform=mobile_transform();encode=model
    model.requires_grad_(False);start=time.perf_counter();torch.cuda.reset_peak_memory_stats();completed=[]
    for scene in cfg['scenes']:
        for seq in manifest['subsplits'][scene]['downstream_fit']+manifest['subsplits'][scene]['normal_calibration']:
            target=ART/args.arm/'features'/scene/f'training_{seq}.npz'
            if target.exists():raise RuntimeError('Refuse to overwrite features: '+str(target))
            selected=[r for r in rows if r['scene']==scene and r['sequence']==seq];ds=ViewDataset(selected,transform);loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=0);features=[]
            with torch.inference_mode():
                for images,idx in loader:
                    z=encode(images.cuda())
                    z=torch.nn.functional.normalize(z.float(),dim=-1);features.append(z.cpu().numpy())
            z=np.concatenate(features);source=Path(cfg['source_features'][scene])/target.name
            assert sha(source)==manifest['normal_source_feature_sha256'][str(source)]
            with np.load(source,allow_pickle=False) as archive:data=dict(archive)
            global_n=len(data['indices']);assert [r['kind'] for r in selected]==['global']*global_n+['crop']*len(data['boxes'])
            data['global_features']=z[:global_n];data['crop_features']=z[global_n:];target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**data)
            completed.append({'scene':scene,'sequence':seq,'views':len(z),'sha256':sha(target)})
            print(args.arm,scene,seq,len(z),flush=True)
    torch.cuda.synchronize();write(OUT/f'{args.arm}_normal_extraction.json',{'arm':args.arm,'normal_only':True,'views':sum(r['views'] for r in completed),'sequences':completed,'elapsed_seconds':time.perf_counter()-start,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'precision':'float32_forward_and_normalization','timing_scope':'model already loaded; includes sequential image IO/preprocessing/GPU/feature serialization; not camera streaming'})

if __name__=='__main__':main()
