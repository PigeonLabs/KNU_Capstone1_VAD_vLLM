"""Label-blind feature extraction. Test annotations are never opened here."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, CLIPModel, CLIPProcessor
from ipad_vad.data import frames_in_order, sample_indices
from ipad_vad.tracking import Tracker


def role_for_phrase(phrase, prompts):
    words=set(phrase.lower().split());scores=[]
    for prompt in prompts:
        tokens=set(prompt.lower().split());scores.append(len(words&tokens)/max(len(words|tokens),1))
    return int(np.argmax(scores)) if max(scores,default=0)>0 else -1


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/experiment01.json'))
    p.add_argument('--limit-sequences',type=int,default=None)
    args=p.parse_args();cfg=json.loads(args.config.read_text());scene=cfg['scene']
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; no silent CPU substitution')
    process=json.loads(Path('results/experiment01/process_discovery.json').read_text())['process']
    paths=json.loads(Path('artifacts/model_paths.json').read_text())
    signature=hashlib.sha256(json.dumps({'config':cfg,'process':process,'models':{k:v['revision'] for k,v in paths.items()}},sort_keys=True).encode()).hexdigest()
    detector_path=paths[cfg['detector']]['local_path'];encoder_path=paths[cfg['encoder']]['local_path']
    detector_processor=AutoProcessor.from_pretrained(detector_path,local_files_only=True)
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(detector_path,local_files_only=True).eval().cuda()
    clip_processor=CLIPProcessor.from_pretrained(encoder_path,local_files_only=True)
    clip=CLIPModel.from_pretrained(encoder_path,local_files_only=True).eval().cuda()
    for model in (detector,clip):
        for parameter in model.parameters():parameter.requires_grad_(False)
    prompts=[o['detection_prompt'] for o in process['objects']]
    text='. '.join(prompts)+'.'
    text_inputs=clip_processor(text=[s['description'] for s in process['phases']],padding=True,return_tensors='pt').to('cuda')
    with torch.inference_mode():
        phase_text=clip.get_text_features(**text_inputs);phase_text=phase_text/phase_text.norm(dim=-1,keepdim=True)
    root=Path('artifacts/experiment01/features')/scene;root.mkdir(parents=True,exist_ok=True)
    logs=[];seen=0
    for partition in ('training','testing'):
        for seq in sorted((args.data_root/scene/partition/'frames').iterdir()):
            if not seq.is_dir():continue
            out=root/f'{partition}_{seq.name}.npz';metadata=out.with_suffix('.json')
            if metadata.exists() and out.exists():
                saved=json.loads(metadata.read_text())
                if saved['signature']!=signature:raise ValueError('Cache signature differs; use a new experiment directory')
                logs.append(saved);continue
            if args.limit_sequences is not None and seen>=args.limit_sequences:break
            files=frames_in_order(seq);indices=sample_indices(len(files),cfg['stride_frames'])
            tracker=Tracker(cfg['tracker_iou_threshold'],cfg['tracker_max_age_samples'])
            global_features=[];logits=[];crop_features=[];object_frames=[];boxes_all=[];roles_all=[];tracks_all=[];conf_all=[]
            unmatched=0;start=time.perf_counter()
            for step,index in enumerate(indices):
                with Image.open(files[index]) as source: image=source.convert('RGB')
                width,height=image.size
                inputs=detector_processor(images=image,text=text,return_tensors='pt').to('cuda')
                with torch.inference_mode(): outputs=detector(**inputs)
                result=detector_processor.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=cfg['detector_threshold'],text_threshold=cfg['text_threshold'],target_sizes=[(height,width)])[0]
                boxes=result['boxes'].detach().cpu().numpy();scores=result['scores'].detach().cpu().numpy()
                phrases=result.get('text_labels',result.get('labels'))
                roles=np.array([role_for_phrase(s,prompts) for s in phrases],dtype=int)
                unmatched+=int((roles<0).sum());keep=[]
                boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,width);boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,height)
                for role in range(len(prompts)):
                    ids=np.flatnonzero((roles==role)&((boxes[:,2]-boxes[:,0])>=2)&((boxes[:,3]-boxes[:,1])>=2))
                    if len(ids):
                        selected=nms(torch.from_numpy(boxes[ids]),torch.from_numpy(scores[ids]),.5).numpy()
                        keep.extend(ids[selected[:3]].tolist())
                boxes,scores,roles=boxes[keep],scores[keep],roles[keep]
                track_ids=tracker.update(boxes,roles,step)
                images=[image]+[image.crop(tuple(map(float,b))) for b in boxes]
                pixels=clip_processor(images=images,return_tensors='pt')['pixel_values'].cuda()
                with torch.inference_mode():
                    features=clip.get_image_features(pixel_values=pixels);features=features/features.norm(dim=-1,keepdim=True)
                    phase_logits=(features[:1]@phase_text.T).cpu().numpy()[0]
                feature_array=features.cpu().numpy();global_features.append(feature_array[0]);logits.append(phase_logits)
                for j,box in enumerate(boxes):
                    crop_features.append(feature_array[j+1]);object_frames.append(step);roles_all.append(roles[j]);tracks_all.append(track_ids[j]);conf_all.append(scores[j])
                    boxes_all.append(box/np.array([width,height,width,height]))
                if step%100==0:print(f'{partition}/{seq.name} sample {step}/{len(indices)} objects={len(boxes)}',flush=True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            phase_sim=np.array(logits);smoothed=np.array([phase_sim[max(0,i-cfg['phase_smoothing_samples']+1):i+1].mean(0) for i in range(len(indices))])
            phases=smoothed.argmax(1)
            np.savez_compressed(out,indices=indices,frame_count=np.array(len(files)),global_features=np.array(global_features),
                                phase_similarity=phase_sim,phases=phases,crop_features=np.array(crop_features,dtype=np.float32).reshape(-1,512),
                                object_frames=np.array(object_frames,dtype=int),roles=np.array(roles_all,dtype=int),tracks=np.array(tracks_all,dtype=int),
                                boxes=np.array(boxes_all,dtype=np.float32).reshape(-1,4),confidence=np.array(conf_all))
            saved={'scene':scene,'partition':partition,'sequence':seq.name,'signature':signature,'source_frames':len(files),
                   'sampled_frames':len(indices),'detected_boxes':len(crop_features),'tracks_created':tracker.next_id,
                   'unmatched_text_detections':unmatched,'elapsed_seconds':elapsed,'phase_counts':np.bincount(phases,minlength=len(process['phases'])).tolist(),
                   'role_frame_coverage':[len(set(np.array(object_frames)[np.array(roles_all)==r].tolist()))/len(indices) for r in range(len(prompts))],
                   'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                   'timing_note':'Sequential sampled-frame extraction only; includes detector, tracker, CLIP and image IO; excludes model load, VLM discovery, fit and final scoring. Not camera streaming throughput.'}
            metadata.write_text(json.dumps(saved,indent=2)+'\n');logs.append(saved);seen+=1
            print(json.dumps(saved),flush=True)
    out=Path('results/experiment01');out.mkdir(parents=True,exist_ok=True)
    (out/'extraction.json').write_text(json.dumps({'sequences':logs,'model_signature':signature},indent=2)+'\n')


if __name__=='__main__':main()
