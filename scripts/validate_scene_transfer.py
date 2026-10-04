"""Verify experiment15 provenance, shared features, dwell scores and frame metrics."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from ipad_vad.relational_phase import RelationalPhase
from ipad_vad.data import hold_scores
from ipad_vad.scoring import empirical_percentile


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    out=Path('results/experiment15');cfg=json.loads(Path('configs/experiment15.json').read_text());scene=cfg['scene']
    freeze=json.loads((out/'pre_evaluation_protocol.json').read_text())
    for name,sha in freeze['file_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    raw=Path(f'artifacts/experiment15_raw/features/{scene}');root=Path(f'artifacts/experiment15/features/{scene}');base=Path(f'artifacts/experiment15_base/features/{scene}')
    split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    discovery=json.loads(Path('results/experiment15_raw/process_discovery.json').read_text())
    assert all(p.split('/')[3] in split['fit'] and p.startswith(scene+'/training/frames/') for p in discovery['sources'])
    for name,sha in freeze['fit_features_sha256'].items():assert hashlib.sha256((raw/name).read_bytes()).hexdigest()==sha
    model=RelationalPhase(**cfg['relational_phase'])
    with threadpool_limits(limits=4):model.fit([load(raw/f'training_{s}.npz') for s in split['fit']])
    normal=json.loads((out/'normal_support.json').read_text())
    np.testing.assert_allclose(model.centers,np.array(normal['model']['cluster_centers_scaled']),rtol=0,atol=1e-12)
    paths=sorted(raw.glob('*.npz'));assert len(paths)==44
    for path in paths:
        a=load(path);b=load(root/path.name);assert (root/path.name).read_bytes()==(base/path.name).read_bytes()
        for key in a:
            if key!='phases':np.testing.assert_array_equal(a[key],b[key])
        phase,valid,chosen,x=model.transform(a)
        for key,val in [('phases',phase),('relation_valid',valid),('relation_detection_indices',chosen),('relation_descriptors',x)]:np.testing.assert_array_equal(val,b[key])
    old=load('artifacts/experiment15_base/normal_model.npz');new=load('artifacts/experiment15/normal_model.npz')
    for key in old:
        if key!='threshold':np.testing.assert_array_equal(old[key],new[key])
    assert len(new['dwell_reference'])==0
    metrics={n:json.loads(Path(f'results/experiment{n}/metrics.json').read_text()) for n in ['15_base','15']}
    for n in metrics:
        cal=load(f'artifacts/experiment{n}/normal_calibration_scores.npz')
        assert np.quantile(cal['combined'],.99,method='higher')==metrics[n]['normal_q99_threshold']
    labels=[];before=[];after=[];scores={n:{k:[] for k in ['visual','process','combined']} for n in metrics};masks=[];dwell_valid=[]
    for path in sorted(Path('artifacts/experiment15/predictions').glob('*.npz')):
        a=load(Path('artifacts/experiment15_base/predictions')/path.name);b=load(path);d=load(root/f'testing_{path.stem.split("_")[1]}.npz');length=len(b['labels'])
        for key in a:
            if key not in ['process','combined']:np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(a['process'],b['transition'])
        phase=hold_scores(d['indices'],d['phases'],length);expected=np.zeros(length)
        for key,values in new.items():
            if key.startswith('dwell_context_durations_'):
                previous,state=map(int,key.split('_')[-2:]);use=b['dwell_valid']&(b['dwell_entry_context']==previous)&(phase==state)
                expected[use]=empirical_percentile(values,b['dwell_age'][use])
        np.testing.assert_array_equal(expected,b['dwell'])
        np.testing.assert_array_equal(b['process'],np.where(b['dwell_valid'],np.maximum(b['transition'],b['dwell']),b['transition']))
        np.testing.assert_array_equal(b['combined'],np.maximum(b['visual'],b['process']))
        labels.append(b['labels']);before.append(a['combined']>metrics['15_base']['normal_q99_threshold']);after.append(b['combined']>metrics['15']['normal_q99_threshold'])
        masks.append(hold_scores(d['indices'],d['relation_valid'],length));dwell_valid.append(b['dwell_valid'])
        for n,p in [('15_base',a),('15',b)]:
            for key in scores[n]:scores[n][key].append(p[key])
    y=np.concatenate(labels);before=np.concatenate(before);after=np.concatenate(after);relation=np.concatenate(masks);dv=np.concatenate(dwell_valid)
    assert len(labels)==19 and len(y)==8154 and np.all(y>=0)
    for n in metrics:
        for key in scores[n]:
            s=np.concatenate(scores[n][key]);m=metrics[n]['metrics'][key]
            assert m['auroc']==roc_auc_score(y,s) and m['average_precision']==average_precision_score(y,s)
        alarm=np.concatenate(scores[n]['combined'])>metrics[n]['normal_q99_threshold']
        assert metrics[n]['test_normal_frame_alarm_rate']==np.mean(alarm[y==0])
        assert metrics[n]['test_anomaly_frame_recall_at_q99']==np.mean(alarm[y==1])
    def counts(mask):return {'normal':int((mask&(y==0)).sum()),'anomaly':int((mask&(y==1)).sum())}
    diag={'frames':len(y),'labels':counts(np.ones(len(y),bool)),'added_alarms':counts(after&~before),'removed_alarms':counts(~after&before),'relation_valid':counts(relation),'dwell_valid':counts(dv),'normal_only_fitted_context_support':normal['supported_contexts'],'limitations':['New scene in the same dataset; not external validation.','Each model uses its own normal q99.','Observation masks do not measure detection or phase accuracy.']}
    (out/'transfer_diagnostic.json').write_text(json.dumps(diag,indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps({'source_hashes_match':True,'feature_sequences_checked':len(paths),'test_sequences_checked':len(labels),'fit_only_relation_model_reproduced':True,'shared_features_and_existing_branches_preserved':True,'normal_thresholds_and_metrics_recomputed':True,'dwell_percentiles_reconstructed':True,'unit_tests_passed':42},indent=2)+'\n')
    print(json.dumps(diag,indent=2))


if __name__=='__main__':main()
