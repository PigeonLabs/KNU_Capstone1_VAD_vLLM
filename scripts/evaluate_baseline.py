"""Fit normal-only models, calibrate on held-out normal videos, evaluate once."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from ipad_vad.data import evaluation_labels,hold_scores
from ipad_vad.scoring import Baseline


def metrics(labels,scores):
    valid=labels>=0;y=labels[valid];s=scores[valid]
    if not np.isfinite(s).all():raise ValueError('Non-finite score')
    return {'frames':len(y),'positive_frames':int(y.sum()),
            'auroc':float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,
            'average_precision':float(average_precision_score(y,s)) if np.any(y==1) else None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/experiment01.json'))
    args=p.parse_args();cfg=json.loads(args.config.read_text());scene=cfg['scene']
    out=Path('results/experiment01');process=json.loads((out/'process_discovery.json').read_text())['process']
    split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    root=Path('artifacts/experiment01/features')/scene
    def load(part,seq):
        with np.load(root/f'{part}_{seq}.npz',allow_pickle=False) as f:return dict(f)
    fit=[load('training',seq) for seq in split['fit']]
    cal=[load('training',seq) for seq in split['calibration']]
    model=Baseline(cfg,process)
    with threadpool_limits(limits=4):
        model.fit(fit);model.calibrate(cal)
        predictions=[];all_labels=[];all_scores={k:[] for k in ['visual','process','combined']};rows=[]
        cache_out=Path('artifacts/experiment01/predictions');cache_out.mkdir(parents=True,exist_ok=True)
        for seq in sorted((args.data_root/scene/'testing/frames').iterdir()):
            if not seq.is_dir():continue
            data=load('testing',seq.name)
            # Labels are first opened AFTER score computation.
            result=model.score(data)
            labels=evaluation_labels(np.load(args.data_root/scene/'test_label'/f'{int(seq.name):03}.npy'),int(data['frame_count']))
            all_labels.append(labels);saved={'labels':labels,'indices':data['indices'],'phases':data['phases']}
            row={'sequence':seq.name,'frames':len(labels),'positive_frames':int((labels==1).sum())}
            for branch in all_scores:
                dense=hold_scores(data['indices'],result[branch],len(labels))
                saved[branch]=dense;all_scores[branch].append(dense)
                row[branch+'_auroc']=metrics(labels,dense)['auroc']
            sampled_roles=[];sampled_frames=[];sampled_scores=[];detection_ids=[]
            for role,frames,scores in result['objects']:
                sampled_roles.extend([role]*len(frames));sampled_frames.extend(frames.tolist());sampled_scores.extend(scores.tolist())
                detection_ids.extend([-1]*len(frames) if role==-1 else np.flatnonzero(data['roles']==role).tolist())
            saved.update(object_roles=np.array(sampled_roles),object_sample_indices=np.array(sampled_frames),object_scores=np.array(sampled_scores),object_detection_indices=np.array(detection_ids),boxes=data['boxes'],tracks=data['tracks'],detected_roles=data['roles'],detected_object_frames=data['object_frames'])
            np.savez_compressed(cache_out/f'{scene}_{seq.name}.npz',**saved);rows.append(row)
            predictions.append(saved)
        y=np.concatenate(all_labels);scores={k:np.concatenate(v) for k,v in all_scores.items()}
        result={'scene':scene,'scope':cfg['scope'],'seed':cfg['seed'],'fit_sequences':split['fit'],'calibration_sequences':split['calibration'],
                'test_sequences':len(rows),'metrics':{k:metrics(y,v) for k,v in scores.items()},'normal_q99_threshold':model.threshold,
                'normal_calibration_sample_alarm_rate':float(np.mean(np.concatenate([model.score(c)['combined'] for c in cal])>model.threshold)),
                'test_normal_frame_alarm_rate':float(np.mean(scores['combined'][y==0]>model.threshold)),
                'test_anomaly_frame_recall_at_q99':float(np.mean(scores['combined'][y==1]>model.threshold)),
                'phase_fit_counts':np.bincount(np.concatenate([d['phases'] for d in fit]),minlength=model.k).tolist(),
                'phase_calibration_counts':np.bincount(np.concatenate([d['phases'] for d in cal]),minlength=model.k).tolist(),
                'phase_test_counts':np.bincount(np.concatenate([p['phases'] for p in predictions]),minlength=model.k).tolist(),
                'subspaces':[{'role':k[0],'phase':k[1],'samples':v.n,'rank':v.rank} for k,v in sorted(model.spaces.items())],
                'transition_probabilities':model.transition.tolist(),
                'limitations':['Single R01 scene, single split/seed; not full IPAD.','No phase/object ground truth; no localization accuracy claimed.',
                               'Geometry/trajectory recorded but not used in anomaly scores in experiment 01.',
                               'Missed objects have no crop score; global branch remains.','PCA fallback pools normal phases if support is insufficient.',
                               'CLIP semantic phase assignment is an unverified proxy.','Frame-level AUPRC is reported as average precision (step integral).',
                               'Q99 alarm uses strict >; no per-test-video normalization.']}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'per_sequence.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    # Normal model artifacts are locally persisted to make inference reproducible.
    arrays={'transition':model.transition,'process_reference':model.process_reference,'threshold':np.array(model.threshold)}
    for (role,phase),space in model.spaces.items():
        arrays[f'mean_{role}_{phase}']=space.mean;arrays[f'basis_{role}_{phase}']=space.basis
    for role,reference in model.calibration.items():arrays[f'calibration_{role}']=reference
    np.savez_compressed('artifacts/experiment01/normal_model.npz',**arrays)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
