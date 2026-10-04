"""Verify gate lineage, causal candidate constraints and refitted normal scoring."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from ipad_vad.verified_anchor import VerifiedAnchorPhase
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from ipad_vad.data import hold_scores


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    out=Path('results/experiment17');cfg=json.loads(Path('configs/experiment17.json').read_text());split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];opt=cfg['anchor_verification']
    pre_margin=json.loads((out/'pre_margin_protocol.json').read_text());assert hashlib.sha256(Path('configs/experiment17.json').read_bytes()).hexdigest()==pre_margin['config_sha256']
    freeze=json.loads((out/'pre_evaluation_protocol.json').read_text());checkpoint=json.loads((out/'pre_test_checkpoint.json').read_text())
    for record in [freeze,checkpoint]:
        for name,sha in record['file_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    source=Path('artifacts/experiment16/features/R04');target=Path('artifacts/experiment17/features/R04');text=load(opt['text_features'])['text_features']
    for name,sha in freeze['source_features_sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha
    phase_model=VerifiedAnchorPhase(text,margin_threshold=opt['margin_threshold'],**cfg['relational_phase'])
    with threadpool_limits(limits=4):phase_model.fit([load(source/f'training_{s}.npz') for s in split['fit']])
    original=load('artifacts/experiment17/normal_relation_model.npz')
    np.testing.assert_allclose(phase_model.centers,original['centers'],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(phase_model.location,original['location']);np.testing.assert_array_equal(phase_model.scale,original['scale'])
    groups={};feature_paths=sorted(source.glob('*.npz'));assert len(feature_paths)==44
    for path in feature_paths:
        a=load(path);b=load(target/path.name);phase,valid,chosen,x=phase_model.transform(a);margin=phase_model.margins(a)
        for key in a:
            if key not in ['phases','relation_valid','relation_detection_indices','relation_descriptors']:np.testing.assert_array_equal(a[key],b[key])
        for key,value in [('phases',phase),('relation_valid',valid),('relation_detection_indices',chosen),('relation_descriptors',x),('anchor_margin',margin)]:np.testing.assert_array_equal(value,b[key])
        ids=chosen[:,0];ids=ids[ids>=0];assert np.all(a['roles'][ids]==phase_model.anchor_role) and np.all(margin[ids]>0)
        part,seq=path.stem.split('_');group='test' if part=='testing' else 'fit' if seq in split['fit'] else 'calibration'
        row=groups.setdefault(group,{'samples':0,'relations_before':0,'relations_after':0,'new_relations':0,'lost_relations':0,'anchor_boxes_before':0,'anchor_boxes_after':0})
        row['samples']+=len(valid);row['relations_before']+=int(a['relation_valid'].sum());row['relations_after']+=int(valid.sum());row['new_relations']+=int(np.sum(valid&~a['relation_valid']));row['lost_relations']+=int(np.sum(~valid&a['relation_valid']));role=a['roles']==phase_model.anchor_role;row['anchor_boxes_before']+=int(role.sum());row['anchor_boxes_after']+=int(np.sum(role&(margin>0)))
    # Refit on normal data and reproduce final calibration/predictions.
    fit=[load(target/f'training_{s}.npz') for s in split['fit']];cal=[load(target/f'training_{s}.npz') for s in split['calibration']]
    with threadpool_limits(limits=4):
        model=LognormalDwellBaseline(cfg,load_process(cfg));model.fit(fit);model.calibrate(cal)
        old_metrics=json.loads(Path('results/experiment16/metrics.json').read_text());new_metrics=json.loads((out/'metrics.json').read_text());assert model.threshold==new_metrics['normal_q99_threshold']
        pre=load('artifacts/experiment17/preflight_calibration_scores.npz');saved_cal=load('artifacts/experiment17/normal_calibration_scores.npz')
        for key in pre:np.testing.assert_array_equal(pre[key],saved_cal[key])
        for key in ['visual','transition','dwell','process','combined','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']:
            np.testing.assert_array_equal(np.concatenate([model.score(d)[key] for d in cal]),saved_cal[key])
        labels=[];before=[];after=[];scores={k:[] for k in ['visual','process','combined']};new_valid=[];old_valid=[]
        pred_paths=sorted(Path('artifacts/experiment17/predictions').glob('*.npz'))
        for path in pred_paths:
            a=load(Path('artifacts/experiment16/predictions')/path.name);b=load(path);d=load(target/f'testing_{path.stem.split("_")[1]}.npz');r=model.score(d);n=len(b['labels'])
            for key in ['labels','indices','boxes','tracks','detected_roles','detected_object_frames','object_roles','object_sample_indices','object_detection_indices']:np.testing.assert_array_equal(a[key],b[key])
            for key in ['visual','transition','dwell','process','combined','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']:np.testing.assert_array_equal(hold_scores(d['indices'],r[key],n),b[key])
            for key in scores:scores[key].append(b[key])
            labels.append(b['labels']);before.append(a['combined']>old_metrics['normal_q99_threshold']);after.append(b['combined']>model.threshold);old_valid.append(a['dwell_valid']);new_valid.append(b['dwell_valid'])
    y=np.concatenate(labels);before=np.concatenate(before);after=np.concatenate(after);old_valid=np.concatenate(old_valid);new_valid=np.concatenate(new_valid)
    assert len(pred_paths)==19 and len(y)==8154
    for key,parts in scores.items():
        values=np.concatenate(parts);assert np.isfinite(values).all();assert roc_auc_score(y,values)==new_metrics['metrics'][key]['auroc'] and average_precision_score(y,values)==new_metrics['metrics'][key]['average_precision']
    assert np.mean(after[y==0])==new_metrics['test_normal_frame_alarm_rate'] and np.mean(after[y==1])==new_metrics['test_anomaly_frame_recall_at_q99']
    def counts(mask):return {'normal':int(np.sum(mask&(y==0))),'anomaly':int(np.sum(mask&(y==1)))}
    result={'groups':groups,'added_alarms':counts(after&~before),'removed_alarms':counts(before&~after),'dwell_valid_before':counts(old_valid),'dwell_valid_after':counts(new_valid),'q99_before':old_metrics['normal_q99_threshold'],'q99_after':model.threshold,'note':'Normal area gates and phase-conditioned models refitted after candidate filter. Appearance inputs unchanged but scores may change. No matched-missingness control; cannot isolate semantic rejection from coverage effects.'}
    (out/'gate_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps({'feature_sequences_checked':44,'test_sequences_checked':19,'protocol_hashes_match':True,'normal_relation_model_reproduced':True,'raw_features_detections_tracks_preserved':True,'every_selected_anchor_passes_frozen_margin':True,'refitted_normal_scores_and_q99_reproduced':True,'preflight_and_final_calibration_identical':True,'metrics_recomputed':True},indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
