"""Experiment 03: re-detect product within normal-motion ROI; preserve other branches."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits
from torchvision.ops import nms
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection,CLIPModel,CLIPProcessor
from ipad_vad.data import frames_in_order
from ipad_vad.tracking import Tracker
from ipad_vad.spatial_phase import SpatialPhase
from ipad_vad.product_roi import fit_product_roi,pixel_roi,restore_boxes
from extract_features import role_for_phrase


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);args=p.parse_args()
    cfg=json.loads(Path('configs/experiment03.json').read_text());scene=cfg['scene']
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    source=Path('artifacts/experiment01/features')/scene
    previous=Path('artifacts/experiment02/features')/scene
    target=Path('artifacts/experiment03/features')/scene;target.mkdir(parents=True,exist_ok=True)
    out=Path('results/experiment03');out.mkdir(parents=True,exist_ok=True)
    split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    fit=[load(source/f'training_{s}.npz') for s in split['fit']]
    roi,evidence=fit_product_roi(fit,**cfg['product_roi'])
    # Freeze the experiment 02 phase map: only its product observations change.
    phase=SpatialPhase(**cfg['spatial_phase'])
    with threadpool_limits(limits=4):phase.fit(fit)
    process=json.loads(Path('results/experiment01/process_discovery.json').read_text())['process']
    prompts=[o['detection_prompt'] for o in process['objects']];text='. '.join(prompts)+'.'
    paths=json.loads(Path('artifacts/model_paths.json').read_text())
    signature=hashlib.sha256(json.dumps({'config':cfg,'process':process,'models':{k:v['revision'] for k,v in paths.items()}},sort_keys=True).encode()).hexdigest()
    detector_processor=AutoProcessor.from_pretrained(paths[cfg['detector']]['local_path'],local_files_only=True)
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(paths[cfg['detector']]['local_path'],local_files_only=True).eval().cuda()
    clip_processor=CLIPProcessor.from_pretrained(paths[cfg['encoder']]['local_path'],local_files_only=True)
    clip=CLIPModel.from_pretrained(paths[cfg['encoder']]['local_path'],local_files_only=True).eval().cuda()
    logs=[]
    for path in sorted(source.glob('*.npz')):
        saved_path=target/path.name;meta=saved_path.with_suffix('.json')
        source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        if saved_path.exists() and meta.exists():
            saved=json.loads(meta.read_text())
            if saved['signature']!=signature or saved['source_sha256']!=source_hash:raise ValueError('Stale cache')
            logs.append(saved);continue
        data=load(path);old=load(previous/path.name);part,seq=path.stem.split('_')
        files=frames_in_order(args.data_root/scene/part/'frames'/seq)
        tracker=Tracker(cfg['tracker_iou_threshold'],cfg['tracker_max_age_samples'])
        # Reserve disjoint IDs; non-product identities and all their features remain unchanged.
        tracker.next_id=int(data['tracks'].max(initial=-1))+1
        product={k:[] for k in ('boxes','confidence','tracks','object_frames','crop_features')}
        start=time.perf_counter()
        for step,index in enumerate(data['indices']):
            with Image.open(files[index]) as im:image=im.convert('RGB')
            width,height=image.size;rect=pixel_roi(roi,width,height);crop=image.crop(tuple(rect))
            inputs=detector_processor(images=crop,text=text,return_tensors='pt').to('cuda')
            with torch.inference_mode():outputs=detector(**inputs)
            result=detector_processor.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=cfg['detector_threshold'],text_threshold=cfg['text_threshold'],target_sizes=[(crop.height,crop.width)])[0]
            boxes=result['boxes'].cpu().numpy();scores=result['scores'].cpu().numpy()
            roles=np.array([role_for_phrase(s,prompts) for s in result.get('text_labels',result.get('labels'))])
            boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,crop.width);boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,crop.height)
            ids=np.flatnonzero((roles==0)&((boxes[:,2]-boxes[:,0])>=2)&((boxes[:,3]-boxes[:,1])>=2))
            if len(ids):ids=ids[nms(torch.from_numpy(boxes[ids]),torch.from_numpy(scores[ids]),.5).numpy()[:3]]
            boxes=restore_boxes(boxes[ids],rect,width,height);scores=scores[ids]
            tracks=tracker.update(boxes,np.zeros(len(boxes),dtype=int),step)
            if len(boxes):
                pixels=clip_processor(images=[image.crop(tuple(map(float,b))) for b in boxes],return_tensors='pt')['pixel_values'].cuda()
                with torch.inference_mode():
                    feat=clip.get_image_features(pixel_values=pixels);feat=feat/feat.norm(dim=-1,keepdim=True)
                product['crop_features'].extend(feat.cpu().numpy())
                product['boxes'].extend(boxes/np.array([width,height,width,height]));product['confidence'].extend(scores)
                product['tracks'].extend(tracks);product['object_frames'].extend([step]*len(boxes))
        keep=data['roles']!=0;original={k:v.copy() for k,v in data.items()}
        for k in product:
            shape=(0,512) if k=='crop_features' else (0,4) if k=='boxes' else (0,)
            new=np.asarray(product[k],dtype=data[k].dtype).reshape((-1,)+shape[1:])
            data[k]=np.concatenate([data[k][keep],new])
        data['roles']=np.r_[data['roles'][keep],np.zeros(len(product['boxes']),dtype=int)]
        data['phases'],observed,_,_=phase.transform(data)
        np.savez_compressed(saved_path,**data)
        for k in original:
            if k not in (*product.keys(),'roles','phases'):assert np.array_equal(data[k],original[k]),k
        for k in (*product.keys(),'roles'):assert np.array_equal(data[k][data['roles']!=0],original[k][keep]),k
        torch.cuda.synchronize()
        saved={'sequence_key':path.stem,'signature':signature,'source_sha256':source_hash,'sampled_frames':len(data['indices']),
               'product_boxes':len(product['boxes']),'direct_bbox_observations':int(observed.sum()),
               'phase_changed_samples':int((data['phases']!=old['phases']).sum()),'elapsed_seconds':time.perf_counter()-start}
        meta.write_text(json.dumps(saved,indent=2)+'\n');logs.append(saved);print(json.dumps(saved),flush=True)
    result={'experiment':'03','roi_model':evidence,'frozen_phase_model':phase.evidence,'fit_sequences':split['fit'],
            'detector_text':text,'signature':signature,'sequences':logs,'test_labels_used':False,
            'unchanged':['non-product boxes, features, track IDs','global features','detector prompt and weights','phase spatial map','score/calibration rules'],
            'timing_scope':'Additional ROI detector + product CLIP + image IO only; excludes original feature extraction and final fit/scoring.',
            'limitations':['ROI may miss off-path anomalies; full-frame branch retained.','ROI resize changes detector input scale and context together.','No independently annotated bbox ground truth.']}
    (out/'grounding.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
