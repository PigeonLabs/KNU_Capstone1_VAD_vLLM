"""Normal holdout observation diagnostics and weak-annotation agreement, never detector GT accuracy."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from ipad_vad.learned_detector import write,sha
from ipad_vad.tracking import iou_matrix
from experiment41_phases import load_npz


def max_missing(mask):
    best=run=0
    for observed in mask:
        run=0 if observed else run+1;best=max(best,run)
    return best


def main():
    cfg=json.loads(Path('configs/experiment40_representation.json').read_text());manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text());rows=[];phase_rows=[]
    for scene in cfg['scenes']:
        for group in ['downstream_fit','normal_calibration']:
            for seq in manifest['subsplits'][scene][group]:
                for arm,root in [('frozen',Path(cfg['source_features'][scene])),('learned',Path('artifacts/experiment41/detections/features')/scene)]:
                    d=load_npz(root/f'training_{seq}.npz');n=len(d['indices'])
                    phase_rows.append({'scene':scene,'group':group,'sequence':seq,'arm':arm,'samples':n,'phase_histogram':{str(int(p)):int(np.sum(d['phases']==p)) for p in np.unique(d['phases'])},'phase_transitions':int(np.sum(d['phases'][1:]!=d['phases'][:-1])),'valid_relations':int(d['relation_valid'].sum()) if 'relation_valid' in d else None})
                    for role in range(2 if scene=='R03' else 3):
                        ids=d['roles']==role;mask=np.isin(np.arange(n),d['object_frames'][ids]);tracks=np.unique(d['tracks'][ids]);jumps=[]
                        for track in tracks:
                            selected=np.flatnonzero(ids&(d['tracks']==track));boxes=d['boxes'][selected];steps=d['object_frames'][selected];center=(boxes[:,:2]+boxes[:,2:])/2
                            if len(center)>1:jumps.extend(np.linalg.norm(np.diff(center,axis=0)[np.diff(steps)==1],axis=1).tolist())
                        rows.append({'scene':scene,'group':group,'sequence':seq,'arm':arm,'role':role,'samples':n,'observed':int(mask.sum()),'max_missing_samples':max_missing(mask),'boxes':int(ids.sum()),'tracks':len(tracks),'mean_within_track_center_step':float(np.mean(jumps)) if jumps else None,'phase_transitions':int(np.sum(d['phases'][1:]!=d['phases'][:-1])),'valid_relations':int(d['relation_valid'].sum()) if 'relation_valid' in d else None})
    annotation=json.loads(Path('results/experiment41/annotations.json').read_text());agreement=[]
    for r in annotation['records']:
        if r['status']!='accepted_weak_annotation' or r['partition']!='representation_validation':continue
        scene,seq=r['scene'],r['sequence']
        for arm,root in [('frozen',Path(cfg['source_features'][scene])),('learned',Path('artifacts/experiment41/detections/features')/scene)]:
            d=load_npz(root/f'training_{seq}.npz');step=r['sample_index'];assert int(d['indices'][step])==r['frame']
            for role in range(2 if scene=='R03' else 3):
                targets=np.array([v['box'] for v in r['annotations'] if v['role']==role],float).reshape(-1,4);pred=d['boxes'][(d['object_frames']==step)&(d['roles']==role)]
                match=[]
                if len(targets) and len(pred):
                    iou=iou_matrix(targets,pred);a,b=linear_sum_assignment(-iou);match=iou[a,b].tolist()
                agreement.append({'scene':scene,'id':r['id'],'arm':arm,'role':role,'weak_target_boxes':len(targets),'predicted_boxes':len(pred),'matched_iou_ge05':sum(v>=.5 for v in match),'assignment_iou_sum':float(sum(match))})
    sums=[]
    for scene in cfg['scenes']:
        for arm in ['frozen','learned']:
            for role in range(2 if scene=='R03' else 3):
                a=[v for v in agreement if v['scene']==scene and v['arm']==arm and v['role']==role];total=sum(v['weak_target_boxes'] for v in a);pred=sum(v['predicted_boxes'] for v in a);matched=sum(v['matched_iou_ge05'] for v in a)
                sums.append({'scene':scene,'arm':arm,'role':role,'weak_target_boxes':total,'predicted_boxes':pred,'matched_iou_ge05':matched,'weak_box_coverage':matched/total if total else None,'weak_box_precision':matched/pred if pred else None})
    write('results/experiment41/normal_diagnostics.json',{'normal_only':True,'per_video_role':rows,'per_video_phase':phase_rows,'weak_validation_agreement':agreement,'weak_agreement_summary':sums,'annotation_sha256':sha('results/experiment41/annotations.json'),'interpretation':'Agreement with model-assisted boxes used for validation selection, not independent mAP/precision/recall; coverage of detector observations is not semantic accuracy. R04 weak validation is mainly upright blade.'})
    print('Normal diagnostics',len(rows),'weak agreement rows',len(agreement))


if __name__=='__main__':main()
