"""Check FIT-duration score reconstruction, unchanged branches and both evaluations."""
import hashlib,json
from pathlib import Path
import numpy as np
from ipad_vad.data import hold_scores
from ipad_vad.scoring import empirical_percentile


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def reconstruct(score,model,phase,context,valid,age):
    expected=np.zeros(len(valid))
    for key,values in model.items():
        if key.startswith('dwell_context_durations_'):
            a,b=map(int,key.split('_')[-2:]);use=valid&(context==a)&(phase==b)
            expected[use]=empirical_percentile(values,age[use])
    np.testing.assert_array_equal(score,expected)


def main():
    out=Path('results/experiment14');cfg=json.loads(Path('configs/experiment14.json').read_text());freeze=json.loads((out/'pre_evaluation_protocol.json').read_text());root=Path('artifacts/experiment14');source=Path('artifacts/experiment12')
    assert hashlib.sha256(Path('configs/experiment14.json').read_bytes()).hexdigest()==freeze['config_sha256']
    for path,sha in freeze['code_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    for name,sha in freeze['source_sha256'].items():
        p=root/'features'/cfg['scene']/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha and p.read_bytes()==(source/'features'/cfg['scene']/name).read_bytes()
    old=load(source/'normal_model.npz');new=load(root/'normal_model.npz')
    assert set(old)==set(new)
    for key in old:
        if key not in ('threshold','dwell_reference'):np.testing.assert_array_equal(old[key],new[key])
    assert len(new['dwell_reference'])==0
    a=load(source/'normal_calibration_scores.npz');b=load(root/'normal_calibration_scores.npz')
    for key in ['visual','transition','dwell_valid','dwell_age','dwell_reason','dwell_entry_context','phases','sequence']:np.testing.assert_array_equal(a[key],b[key])
    reconstruct(b['dwell'],new,b['phases'],b['dwell_entry_context'],b['dwell_valid'],b['dwell_age'])
    np.testing.assert_array_equal(b['combined'],np.maximum(b['visual'],np.where(b['dwell_valid'],np.maximum(b['transition'],b['dwell']),b['transition'])))
    assert float(new['threshold'])==np.quantile(b['combined'],cfg['calibration_quantile'],method='higher')
    rows=[];all_y=[];all_a=[];all_b=[];all_context=[];all_phase=[];all_valid=[];all_dwell=[]
    paths=sorted((source/'predictions').glob('*.npz'));assert [p.name for p in paths]==sorted(p.name for p in (root/'predictions').glob('*.npz'))
    for path in paths:
        a=load(path);b=load(root/'predictions'/path.name);d=load(root/'features'/cfg['scene']/f'testing_{path.stem.split("_")[1]}.npz')
        for key in a:
            if key not in ['process','combined','dwell']:np.testing.assert_array_equal(a[key],b[key])
        phase=hold_scores(d['indices'],d['phases'],len(b['labels']))
        reconstruct(b['dwell'],new,phase,b['dwell_entry_context'],b['dwell_valid'],b['dwell_age'])
        np.testing.assert_array_equal(b['process'],np.where(b['dwell_valid'],np.maximum(b['transition'],b['dwell']),b['transition']))
        np.testing.assert_array_equal(b['combined'],np.maximum(b['visual'],b['process']))
        assert np.isfinite(b['combined']).all()
        all_y.append(b['labels']);all_a.append(a['combined']);all_b.append(b['combined']);all_context.append(b['dwell_entry_context']);all_phase.append(phase);all_valid.append(b['dwell_valid']);all_dwell.append(b['dwell'])
    y=np.concatenate(all_y);before=np.concatenate(all_a)>float(old['threshold']);after=np.concatenate(all_b)>float(new['threshold']);context=np.concatenate(all_context);phase=np.concatenate(all_phase);valid=np.concatenate(all_valid);ds=np.concatenate(all_dwell)
    def counts(mask):return {'normal':int((mask&(y==0)).sum()),'anomaly':int((mask&(y==1)).sum())}
    strata={}
    for entry,state in np.unique(np.column_stack([context,phase]),axis=0):
        use=(context==entry)&(phase==state)
        strata[f'{entry}->{state}']={'frames':counts(use),'before_alarms':counts(use&before),'after_alarms':counts(use&after),'added':counts(use&after&~before),'removed':counts(use&~after&before)}
    bounds=[]
    for key,values in new.items():
        if key.startswith('dwell_context_durations_'):
            a,b=map(int,key.split('_')[-2:]);bounds.append({'context':f'{a}->{b}','fit_complete_runs':len(values),'fit_max_duration':float(values.max()),'max_percentile_within_fit_range':float(empirical_percentile(values,values.max()))})
    normal=json.loads((out/'normal_holdout.json').read_text())
    for row in normal['folds']:
        held=row['held_out'];fold=root/'normal_holdout'/f'holdout_{held}';model=load(fold/'normal_model.npz');scores=load(fold/'heldout_scores.npz');cal=load(fold/'calibration_scores.npz')
        oldfold=Path(f'artifacts/experiment13/variant12/holdout_{held}');oldmodel=load(oldfold/'normal_model.npz');oldscore=load(oldfold/'heldout_scores.npz')
        for key in model:
            if key not in ('threshold','dwell_reference'):np.testing.assert_array_equal(model[key],oldmodel[key])
        for key in ['visual','transition','dwell_valid','dwell_age']:np.testing.assert_array_equal(scores[key],oldscore[key])
        reconstruct(scores['dwell'],model,scores['phases'],scores['dwell_entry_context'],scores['dwell_valid'],scores['dwell_age'])
        np.testing.assert_array_equal(scores['combined'],np.maximum(scores['visual'],np.maximum(scores['transition'],scores['dwell'])))
        assert float(model['threshold'])==np.quantile(cal['combined'],cfg['calibration_quantile'],method='higher')
        dense=hold_scores(scores['indices'],scores['combined'],int(scores['frame_count']));np.testing.assert_array_equal(dense,scores['dense_combined'])
        alarm=dense>float(model['threshold']);np.testing.assert_array_equal(alarm,scores['dense_alarm']);assert row['false_positive_frames']==int(alarm.sum())
    diag={'threshold_before':float(old['threshold']),'threshold_after':float(new['threshold']),'added_alarms':counts(after&~before),'removed_alarms':counts(~after&before),'both_alarms':counts(after&before),
          'valid_dwell_at_one':counts(valid&(ds==1)),'context_strata':strata,'fit_duration_bounds':bounds,
          'note':'Separate normal q99 thresholds. With few complete runs, within-FIT percentiles may be below final q99 even for abnormal observations.'}
    (out/'completed_dwell_diagnostic.json').write_text(json.dumps(diag,indent=2)+'\n')
    val={'feature_sequences_checked':len(freeze['source_sha256']),'test_predictions_checked':len(paths),'normal_holdout_folds_checked':len(normal['folds']),
        'same_features_visual_transition_masks_ages_objects_and_labels':True,'same_fitted_duration_banks':True,'complete_run_scores_reconstructed':True,
        'normal_q99_reconstructed':True,'empty_unused_calibration_age_reference':True,'config_code_source_hashes_match':True,'artifact_validation_only':True}
    (out/'validation.json').write_text(json.dumps(val,indent=2)+'\n');print(json.dumps(diag,indent=2))

if __name__=='__main__':main()
