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
from ipad_vad.experiment import load_process
from ipad_vad.kinematic_scoring import KinematicBaseline
from ipad_vad.process_calibration import StateCalibratedBaseline
from ipad_vad.dwell_scoring import DwellBaseline


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
    experiment='experiment'+cfg['experiment']
    out=Path('results')/experiment;out.mkdir(parents=True,exist_ok=True);process=load_process(cfg)
    split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    root=Path('artifacts')/experiment/'features'/scene
    def load(part,seq):
        with np.load(root/f'{part}_{seq}.npz',allow_pickle=False) as f:return dict(f)
    fit=[load('training',seq) for seq in split['fit']]
    cal=[load('training',seq) for seq in split['calibration']]
    if 'process_calibration' in cfg:
        if 'normal_progress' in cfg:raise ValueError('Conditional process calibration with motion is not specified')
        if cfg['process_calibration']['mode']!='previous_state':raise ValueError('Unknown process calibration mode')
        model=DwellBaseline(cfg,process) if 'normal_dwell' in cfg else StateCalibratedBaseline(cfg,process)
    else:model=KinematicBaseline(cfg,process) if 'normal_progress' in cfg else Baseline(cfg,process)
    if 'normal_dwell' in cfg and 'process_calibration' not in cfg:raise ValueError('Dwell experiment requires previous-state process calibration')
    with threadpool_limits(limits=4):
        model.fit(fit);model.calibrate(cal)
        predictions=[];all_labels=[];all_scores={k:[] for k in ['visual','process','combined']};rows=[]
        cache_out=Path('artifacts')/experiment/'predictions';cache_out.mkdir(parents=True,exist_ok=True)
        for seq in sorted((args.data_root/scene/'testing/frames').iterdir()):
            if not seq.is_dir():continue
            data=load('testing',seq.name)
            # Labels are first opened AFTER score computation.
            result=model.score(data)
            labels=evaluation_labels(np.load(args.data_root/scene/'test_label'/f'{int(seq.name):03}.npy'),int(data['frame_count']))
            all_labels.append(labels);saved={'labels':labels,'indices':data['indices'],'phases':data['phases']}
            row={'sequence':seq.name,'frames':len(labels),'positive_frames':int((labels==1).sum())}
            if 'process_calibration' in cfg:
                saved['process_conditioned']=model.conditioning_mask(data)
            if 'dwell' in result:
                for key in ('transition','dwell','dwell_age','dwell_reason','dwell_valid'):
                    saved[key]=hold_scores(data['indices'],result[key],len(labels))
                row['dwell_valid_frames']=int(saved['dwell_valid'].sum())
            if 'motion' in result:
                saved['motion']=hold_scores(data['indices'],result['motion'],len(labels))
                saved['motion_valid']=hold_scores(data['indices'],result['motion_valid'].astype(float),len(labels)).astype(bool)
                saved['transition']=hold_scores(data['indices'],result['transition'],len(labels))
                saved['motion_detection_index']=data['motion_detection_index']
                row['motion_valid_frames']=int(saved['motion_valid'].sum())
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
        cal_results=[model.score(c) for c in cal]
        cal_scores={key:np.concatenate([r[key] for r in cal_results]) for key in ('visual','process','combined')}
        if 'normal_dwell' in cfg:
            for key in ('transition','dwell','dwell_valid','dwell_age','dwell_reason'):
                cal_scores[key]=np.concatenate([r[key] for r in cal_results])
        np.savez_compressed(Path('artifacts')/experiment/'normal_calibration_scores.npz',**cal_scores,
                            phases=np.concatenate([d['phases'] for d in cal]),
                            sequence=np.concatenate([np.full(len(d['phases']),seq) for seq,d in zip(split['calibration'],cal)]))
        result={'scene':scene,'scope':cfg['scope'],'seed':cfg['seed'],'fit_sequences':split['fit'],'calibration_sequences':split['calibration'],
                'test_sequences':len(rows),'metrics':{k:metrics(y,v) for k,v in scores.items()},'normal_q99_threshold':model.threshold,
                'normal_calibration_sample_alarm_rate':float(np.mean(cal_scores['combined']>model.threshold)),
                'score_fusion':cfg.get('score_fusion','mean'),
                'test_normal_frame_alarm_rate':float(np.mean(scores['combined'][y==0]>model.threshold)),
                'test_anomaly_frame_recall_at_q99':float(np.mean(scores['combined'][y==1]>model.threshold)),
                'phase_fit_counts':np.bincount(np.concatenate([d['phases'] for d in fit]),minlength=model.k).tolist(),
                'phase_calibration_counts':np.bincount(np.concatenate([d['phases'] for d in cal]),minlength=model.k).tolist(),
                'phase_test_counts':np.bincount(np.concatenate([p['phases'] for p in predictions]),minlength=model.k).tolist(),
                'subspaces':[{'role':k[0],'phase':k[1],'samples':v.n,'rank':v.rank} for k,v in sorted(model.spaces.items())],
                'transition_probabilities':model.transition.tolist(),
                'limitations':[f'Single {scene} scene, single split/seed; not full IPAD.','No phase/object ground truth; no localization accuracy claimed.',
                               'Appearance scoring uses CLIP residuals; optional normal geometry/motion branches are specified in the experiment config.',
                               'Missed objects have no crop score; global branch remains.','PCA fallback pools normal phases if support is insufficient.',
                               'Phase assignments are proxies, not ground truth.','Frame-level AUPRC is reported as average precision (step integral).',
                               'Q99 alarm uses strict >; no per-test-video normalization.']}
    if 'normal_progress' in cfg:
        motion=np.concatenate([p['motion'] for p in predictions]);valid=np.concatenate([p['motion_valid'] for p in predictions])
        result['motion_valid_only_metrics']=metrics(np.where(valid,y,-1),motion)
        result['motion_test_valid_fraction']=float(valid.mean())
        result['motion_model']={str(k):v for k,v in model.motion.models.items()}
        result['motion_calibration_samples']=len(model.motion.reference)
        result['limitations'].append('Motion-only metrics cover valid dense intervals only, not the full test set. Invalid motion retains transition score.')
    if 'process_calibration' in cfg:
        result['process_calibration']={'mode':'previous_state','minimum_support':model.minimum_support,
            'normal_transition_support':{str(k):len(v) for k,v in model.state_process_references.items()},
            'test_sampled_conditional_fraction':float(np.concatenate([p['process_conditioned'] for p in predictions]).mean()),
            'fallback':'Global reference for first observation or insufficient previous-state support.'}
    if 'normal_dwell' in cfg:
        dwell=np.concatenate([p['dwell'] for p in predictions]);valid=np.concatenate([p['dwell_valid'] for p in predictions]).astype(bool)
        result['dwell']={'complete_run_support':model.dwell.support,'supported_states':sorted(model.dwell.durations),
            'normal_calibration_samples':len(model.dwell.reference),'test_dense_valid_fraction':float(valid.mean()),
            'valid_only_metrics':metrics(np.where(valid,y,-1),dwell),
            'reason_counts':{str(i):int((np.concatenate([p['dwell_reason'] for p in predictions])==i).sum()) for i in range(4)},
            'reason_codes':{'0':'valid','1':'relation_missing','2':'entry_not_observed','3':'unsupported_state'}}
        result['limitations'].append('Dwell-only metrics cover valid intervals only. Unsupported, missing and unobserved-entry intervals retain the transition branch; cluster duration is not action-duration ground truth.')
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'per_sequence.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    # Normal model artifacts are locally persisted to make inference reproducible.
    arrays={'transition':model.transition,'process_reference':model.process_reference,'threshold':np.array(model.threshold)}
    for (role,phase),space in model.spaces.items():
        arrays[f'mean_{role}_{phase}']=space.mean;arrays[f'basis_{role}_{phase}']=space.basis
    for role,reference in model.calibration.items():arrays[f'calibration_{role}']=reference
    if 'process_calibration' in cfg:
        for state,reference in model.state_process_references.items():arrays[f'process_reference_state_{state}']=reference
    if 'normal_dwell' in cfg:
        arrays['dwell_reference']=model.dwell.reference
        for state,durations in model.dwell.durations.items():arrays[f'dwell_durations_state_{state}']=durations
    if 'normal_progress' in cfg:
        arrays['motion_reference']=model.motion.reference
        for phase,parameters in model.motion.models.items():
            arrays[f'motion_parameters_{phase}']=np.array([parameters['median'],parameters['scale'],parameters['samples']])
    np.savez_compressed(Path('artifacts')/experiment/'normal_model.npz',**arrays)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
