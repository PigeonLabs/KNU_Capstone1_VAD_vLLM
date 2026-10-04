"""Validate the duration-only change and diagnose its alarms on the development set."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from ipad_vad.dwell import NormalDwell
from ipad_vad.data import hold_scores


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',default='10');p.add_argument('--experiment',default='11');args=p.parse_args();n=args.experiment
    cfg=json.loads(Path(f'configs/experiment{n}.json').read_text());scene=cfg['scene'];source=Path(f'artifacts/experiment{args.baseline}');target=Path(f'artifacts/experiment{n}')
    protocol=json.loads(Path(f'results/experiment{n}/pre_evaluation_protocol.json').read_text())
    assert hashlib.sha256(Path(f'configs/experiment{n}.json').read_bytes()).hexdigest()==protocol['config_sha256']
    for path,sha in protocol['code_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    names=sorted(p.name for p in (source/'features'/scene).glob('*.npz'));assert names==sorted(p.name for p in (target/'features'/scene).glob('*.npz'))
    for name in names:
        a=(source/'features'/scene/name).read_bytes();b=(target/'features'/scene/name).read_bytes();assert a==b and hashlib.sha256(b).hexdigest()==protocol['source_sha256'][name]
    old=load(source/'normal_model.npz');new=load(target/'normal_model.npz')
    for key in old:
        if key!='threshold':np.testing.assert_array_equal(old[key],new[key],err_msg=key)
    split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    normal={group:[load(target/'features'/scene/f'training_{seq}.npz') for seq in split[group]] for group in ('fit','calibration')}
    dwell=NormalDwell(**cfg['normal_dwell']);dwell.fit(normal['fit']);dwell.calibrate(normal['calibration'])
    np.testing.assert_array_equal(new['dwell_reference'],dwell.reference)
    for state,values in dwell.durations.items():np.testing.assert_array_equal(new[f'dwell_durations_state_{state}'],values)
    a=load(source/'normal_calibration_scores.npz');b=load(target/'normal_calibration_scores.npz')
    np.testing.assert_array_equal(a['visual'],b['visual']);np.testing.assert_array_equal(a['process'],b['transition'])
    np.testing.assert_array_equal(b['process'],np.where(b['dwell_valid'],np.maximum(b['transition'],b['dwell']),b['transition']))
    np.testing.assert_array_equal(b['combined'],np.maximum(b['visual'],b['process']))
    assert float(new['threshold'])==np.quantile(b['combined'],cfg['calibration_quantile'],method='higher')
    calibration_scores=b['combined']
    summaries=[];all_y=[];all_old=[];all_new=[];all_valid=[];all_phase=[];all_dwell=[];all_visual=[]
    paths=sorted((source/'predictions').glob('*.npz'));assert [p.name for p in paths]==sorted(p.name for p in (target/'predictions').glob('*.npz'))
    for path in paths:
        seq=path.stem.split('_')[1];d=load(target/'features'/scene/f'testing_{seq}.npz');a=load(path);b=load(target/'predictions'/path.name)
        for key in a:
            if key not in ('process','combined'):np.testing.assert_array_equal(a[key],b[key],err_msg=path.name+':'+key)
        np.testing.assert_array_equal(a['process'],b['transition'])
        score,valid,age,reason=dwell.score(d)
        for key,values in [('dwell',score),('dwell_valid',valid),('dwell_age',age),('dwell_reason',reason)]:
            np.testing.assert_array_equal(b[key],hold_scores(d['indices'],values,len(b['labels'])))
        np.testing.assert_array_equal(b['process'],np.where(b['dwell_valid'],np.maximum(b['transition'],b['dwell']),b['transition']))
        np.testing.assert_array_equal(b['combined'],np.maximum(b['visual'],b['process']))
        np.testing.assert_array_equal(a['combined'][~b['dwell_valid']],b['combined'][~b['dwell_valid']])
        assert np.all(b['combined']>=a['combined']) and np.isfinite(b['combined']).all()
        summaries.append({'sequence':seq,'sampled':len(valid),'valid_sampled':int(valid.sum()),'valid_dense':int(b['dwell_valid'].sum())})
        all_y.append(b['labels']);all_old.append(a['combined']);all_new.append(b['combined']);all_valid.append(b['dwell_valid']);all_phase.append(hold_scores(d['indices'],d['phases'],len(b['labels'])));all_dwell.append(b['dwell']);all_visual.append(b['visual'])
    y=np.concatenate(all_y);previous=np.concatenate(all_old);current=np.concatenate(all_new);valid=np.concatenate(all_valid);phases=np.concatenate(all_phase);ds=np.concatenate(all_dwell);visual=np.concatenate(all_visual)
    previous_alarm=previous>float(old['threshold']);alarm=current>float(new['threshold'])
    def counts(mask):return {'normal':int((mask&(y==0)).sum()),'anomaly':int((mask&(y==1)).sum())}
    strata={}
    for name,use in [('valid',valid),('invalid',~valid)]+[(f'state_{i}',phases==i) for i in range(cfg['relational_phase']['k'])]:
        strata[name]={'frames':counts(use),'old_alarms':counts(use&previous_alarm),'new_alarms':counts(use&alarm),
            'added_alarms':counts(use&alarm&~previous_alarm),'lost_alarms':counts(use&~alarm&previous_alarm),'score_increased':counts(use&(current>previous))}
    diag={'scene':scene,'threshold_before':float(old['threshold']),'threshold_after':float(new['threshold']),'strata':strata,
        'dwell_exceeds_old_combined':counts(valid&(ds>previous)),'dwell_test_at_one_valid':counts(valid&(ds==1)),
        'normal_calibration_at_q99':int((calibration_scores==float(new['threshold'])).sum()),
        'normal_calibration_max':float(calibration_scores.max()),'sequences':summaries,
        'note':'Dwell-only metrics use valid intervals; final metrics use all matched test frames. Alarm changes include separately calibrated q99, not equal test FPR.'}
    validation={'feature_sequences_checked':len(names),'test_predictions_checked':len(paths),'features_phases_visual_objects_labels_and_transition_identical':True,
        'shared_normal_model_arrays_except_threshold_identical':True,'normal_only_durations_and_reference_reconstructed':True,
        'invalid_dwell_preserves_old_score':True,'duration_fusion_and_normal_q99_reconstructed':True,'config_and_code_match_pre_evaluation_freeze':True,'artifact_validation_only':True}
    out=Path(f'results/experiment{n}');(out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');(out/'dwell_diagnostic.json').write_text(json.dumps(diag,indent=2)+'\n');print(json.dumps(diag,indent=2))

if __name__=='__main__':main()
