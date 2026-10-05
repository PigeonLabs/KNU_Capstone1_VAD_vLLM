"""Learned detector extraction. Labels never read; frozen CLIP auxiliary path retained."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import CLIPModel,CLIPProcessor
from ipad_vad.data import frames_in_order
from ipad_vad.tracking import Tracker
from ipad_vad.learned_detector import load_detector,configure_detector,restore_adapted,sha,write
from extract_features import role_for_phrase
from experiment41_phases import phase_models,transform,load_npz

OUT=Path('results/experiment41');ART=Path('artifacts/experiment41/detections/features')


def main():
    p=argparse.ArgumentParser();p.add_argument('--partition',choices=['training','testing'],required=True);a=p.parse_args()
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            s=str(args[0]);name=Path(s).name
            if '/test_label/' in s or '/predictions/' in s:raise RuntimeError('Label/prediction access forbidden during extraction')
            if a.partition=='training' and ('/testing/' in s or name.startswith('testing_')):raise RuntimeError('Test access before normal checkpoint')
    sys.addaudithook(guard)
    if a.partition=='testing' and not (OUT/'normal_models_checkpoint.json').exists():raise RuntimeError('Normal models not frozen')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    cfg=json.loads(Path('configs/experiment41_detector.json').read_text());rep=json.loads(Path('configs/experiment40_representation.json').read_text());train=json.loads((OUT/'training.json').read_text())
    assert train['completed'] and sha(train['checkpoint'])==train['checkpoint_sha256']
    model,processor,_=load_detector();configure_detector(model);restore_adapted(model,torch.load(train['checkpoint'],weights_only=True)['state']);model.requires_grad_(False);model.eval().cuda()
    cr=json.loads(Path('artifacts/model_paths.json').read_text())['openai/clip-vit-base-patch32'];clip=CLIPModel.from_pretrained(cr['local_path'],local_files_only=True).eval().cuda();clip.requires_grad_(False);cp=CLIPProcessor.from_pretrained(cr['local_path'],local_files_only=True)
    ms=phase_models();manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text());logs=[];start=time.perf_counter()
    signature=sha(OUT/'training.json')+sha(Path(__file__))+sha('scripts/experiment41_phases.py')
    for scene in rep['scenes']:
        options=json.loads(Path(f'configs/experiment40_A_{scene}.json').read_text());prompts=cfg['prompts'][scene];caption='. '.join(prompts)+'.'
        seqs=(manifest['subsplits'][scene]['downstream_fit']+manifest['subsplits'][scene]['normal_calibration']) if a.partition=='training' else sorted(p.name for p in (Path(cfg['data_root'])/scene/'testing/frames').iterdir() if p.is_dir())
        for seq in seqs:
            dest=ART/scene/f'{a.partition}_{seq}.npz';meta=dest.with_suffix('.json')
            if dest.exists() and meta.exists():
                log=json.loads(meta.read_text());assert log['signature']==signature and log['sha256']==sha(dest);logs.append(log);continue
            original=load_npz(Path(rep['source_features'][scene])/dest.name);files=frames_in_order(Path(cfg['data_root'])/scene/a.partition/'frames'/seq)
            assert len(files)==int(original['frame_count']);tracker=Tracker(options['tracker_iou_threshold'],options['tracker_max_age_samples'])
            box_list=[];roles_list=[];tracks_list=[];confidence=[];steps=[];crop_features=[];begin=time.perf_counter()
            for step,index in enumerate(original['indices']):
                with Image.open(files[index]) as f:im=f.convert('RGB')
                w,h=im.size;inputs=processor(images=im,text=caption,return_tensors='pt').to('cuda')
                with torch.inference_mode():out=model(**inputs)
                result=processor.post_process_grounded_object_detection(out,inputs.input_ids,threshold=options['detector_threshold'],text_threshold=options['text_threshold'],target_sizes=[(h,w)])[0]
                b=result['boxes'].cpu().numpy();sc=result['scores'].cpu().numpy();rr=np.array([role_for_phrase(s,prompts) for s in result.get('text_labels',result.get('labels'))],dtype=int)
                b[:,[0,2]]=np.clip(b[:,[0,2]],0,w);b[:,[1,3]]=np.clip(b[:,[1,3]],0,h);keep=[]
                for role in range(len(prompts)):
                    ids=np.flatnonzero((rr==role)&((b[:,2]-b[:,0])>=2)&((b[:,3]-b[:,1])>=2))
                    if len(ids):keep.extend(ids[nms(torch.from_numpy(b[ids]),torch.from_numpy(sc[ids]),.5).numpy()[:3]].tolist())
                b,sc,rr=b[keep],sc[keep],rr[keep];tt=tracker.update(b,rr,step)
                pixels=cp(images=[im]+[im.crop(tuple(map(float,v))) for v in b],return_tensors='pt')['pixel_values'].cuda()
                with torch.inference_mode():z=clip.get_image_features(pixel_values=pixels);z=z/z.norm(dim=-1,keepdim=True)
                crop_features.extend(z[1:].cpu().numpy());box_list.extend(b/np.array([w,h,w,h]));roles_list.extend(rr);tracks_list.extend(tt);confidence.extend(sc);steps.extend([step]*len(b))
            data={k:original[k] for k in ['indices','frame_count','global_features','phase_similarity','phases']}
            data.update(boxes=np.array(box_list,dtype=np.float32).reshape(-1,4),roles=np.array(roles_list,dtype=int),tracks=np.array(tracks_list,dtype=int),confidence=np.array(confidence),object_frames=np.array(steps,dtype=int),crop_features=np.array(crop_features,dtype=np.float32).reshape(-1,512))
            data=transform(scene,data,ms);dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**data)
            log={'scene':scene,'sequence':seq,'partition':a.partition,'signature':signature,'sha256':sha(dest),'sampled_frames':len(data['indices']),'boxes':len(steps),'tracks':tracker.next_id,'phase_changes_from_frozen':int((data['phases']!=original['phases']).sum()),'role_frame_coverage':[len(set(data['object_frames'][data['roles']==role].tolist()))/len(data['indices']) for role in range(len(prompts))],'phase_counts':np.bincount(data['phases'],minlength=len(options.get('phases',[])) or int(original['phases'].max())+1).tolist(),'valid_relations':int(data['relation_valid'].sum()) if 'relation_valid' in data else None,'elapsed_seconds':time.perf_counter()-begin}
            write(meta,log);logs.append(log);print(json.dumps(log),flush=True)
    write(OUT/f'detector_{a.partition}_extraction.json',{'normal_only':a.partition=='training','sequences':logs,'elapsed_seconds':time.perf_counter()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'labels_read':False,'full_frame_clip_auxiliary_reused':True,'frozen_phase_parameters_reused':True,'timing_scope':'includes model already loaded sequential detector/CLIP crops, image IO, tracking, frozen phase transform and serialization; excludes VLM and model fitting; cache skips excluded from elapsed'})


if __name__=='__main__':main()
