"""Freeze all eight experiment40 encoders; encode only the new detector crops."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from ipad_vad.data import frames_in_order
from ipad_vad.learned_visual import load_mobile_visual,configure_adaptation
from ipad_vad.learned_detector import sha,write
from ipad_vad.representation_data import ViewDataset,mobile_transform
from experiment40_normal import RUNS
from experiment41_phases import load_npz

OUT=Path('results/experiment41');ART=Path('artifacts/experiment41')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',choices=RUNS,required=True);p.add_argument('--partition',choices=['training','testing'],required=True);args=p.parse_args()
    def guard(event,params):
        if event=='open' and isinstance(params[0],(str,bytes)):
            s=str(params[0])
            if '/test_label/' in s or '/predictions/' in s:raise RuntimeError('Labels forbidden')
            if args.partition=='training' and ('/testing/' in s or Path(s).name.startswith('testing_')):raise RuntimeError('Test access in normal extraction')
    sys.addaudithook(guard)
    if args.partition=='testing' and not (OUT/'normal_models_checkpoint.json').exists():raise RuntimeError('Normal models not frozen')
    cfg=json.loads(Path('configs/experiment40_representation.json').read_text());manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text());run=args.run
    detections=json.loads((OUT/f'detector_{args.partition}_extraction.json').read_text());torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if run=='A':
        from transformers import CLIPModel,CLIPProcessor
        record=json.loads(Path('artifacts/model_paths.json').read_text())['openai/clip-vit-base-patch32'];model=CLIPModel.from_pretrained(record['local_path'],local_files_only=True).cuda().eval();processor=CLIPProcessor.from_pretrained(record['local_path'],local_files_only=True).image_processor
        transform=T.Compose([T.Resize(224,interpolation=InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize(processor.image_mean,processor.image_std)]);encode=lambda x:model.get_image_features(pixel_values=x);checkpoint_sha=None
    else:
        model,_=load_mobile_visual(manifest['model']);transform=mobile_transform();checkpoint_sha=None
        if run!='B':
            training=json.loads(Path(f'results/experiment40/{run}_training.json').read_text());checkpoint_sha=training['checkpoint_sha256'];assert sha(training['checkpoint'])==checkpoint_sha
            scope=configure_adaptation(model,'lora' if run.startswith('C') else 'full',cfg['lora_rank'],cfg['lora_alpha']);assert scope==training['scope'];saved=torch.load(training['checkpoint'],map_location='cpu',weights_only=True);assert saved['epoch']==training['selected_epoch'];model.load_state_dict(saved['visual'],strict=True)
        model.cuda().eval();encode=model
    model.requires_grad_(False);start=time.perf_counter();completed=[]
    signature=sha(Path(__file__))+sha(OUT/f'detector_{args.partition}_extraction.json')+(checkpoint_sha or run)
    old_record=json.loads(Path(f'results/experiment40/{run}_{"normal" if args.partition=="training" else "test"}_extraction.json').read_text());old_hashes={(r['scene'],r['sequence']):r['sha256'] for r in old_record['sequences']}
    for row in detections['sequences']:
        scene,seq=row['scene'],row['sequence'];name=f'{args.partition}_{seq}.npz';target=ART/run/'features'/scene/name;meta=target.with_suffix('.json')
        if target.exists() and meta.exists():
            prior=json.loads(meta.read_text());assert prior['signature']==signature and prior['sha256']==sha(target);completed.append(prior);continue
        source=ART/'detections/features'/scene/name;assert sha(source)==row['sha256'];data=load_npz(source);images=frames_in_order(Path(cfg['data_root'])/scene/args.partition/'frames'/seq)
        views=[{'path':str(images[int(data['indices'][step])]),'box':box.tolist()} for step,box in zip(data['object_frames'],data['boxes'])]
        loader=DataLoader(ViewDataset(views,transform),batch_size=64,shuffle=False,num_workers=0);features=[]
        with torch.inference_mode():
            for batch,_ in loader:features.append(torch.nn.functional.normalize(encode(batch.cuda()).float(),dim=-1).cpu().numpy())
        z=np.concatenate(features) if features else np.empty((0,512),np.float32)
        assert np.isfinite(z).all();np.testing.assert_allclose(np.linalg.norm(z,axis=1),1,rtol=1e-5,atol=1e-6)
        old_path=Path('artifacts/experiment40')/run/'features'/scene/name;assert sha(old_path)==old_hashes[(scene,seq)]
        old=load_npz(old_path);np.testing.assert_array_equal(old['indices'],data['indices']);data['global_features']=old['global_features'];data['crop_features']=z
        target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**data)
        info={'scene':scene,'sequence':seq,'partition':args.partition,'signature':signature,'sha256':sha(target),'new_crop_views':len(z),'reused_global_views':len(data['indices']),'global_source_sha256':sha(old_path),'detector_source_sha256':sha(source)};write(meta,info);completed.append(info);print(run,scene,seq,len(z),flush=True)
    write(OUT/f'{run}_{args.partition}_extraction.json',{'run':run,'normal_only':args.partition=='training','labels_opened':False,'encoders_retrained':False,'checkpoint_sha256':checkpoint_sha,'sequences':completed,'elapsed_seconds':time.perf_counter()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'precision':'float32_forward_and_normalization','global_features_reused_exactly_from_40':True})


if __name__=='__main__':main()
