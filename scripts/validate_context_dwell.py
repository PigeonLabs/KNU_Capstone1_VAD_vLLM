"""Verify matched support and isolate entry-duration conditioning from abstention."""
import hashlib,json
from pathlib import Path
import numpy as np
from ipad_vad.dwell import NormalDwell
from ipad_vad.context_dwell import ContextDwell, observed_entry_context
from ipad_vad.data import hold_scores


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    ns=['11','12_support','12'];cfgs={n:json.loads(Path(f'configs/experiment{n}.json').read_text()) for n in ns};scene=cfgs['11']['scene']
    roots={n:Path(f'artifacts/experiment{n}') for n in ns};models={n:load(roots[n]/'normal_model.npz') for n in ns};base=models['11']
    split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    caches={group:[load(roots['11']/'features'/scene/f'training_{seq}.npz') for seq in split[group]] for group in ['fit','calibration']}
    normal=NormalDwell(**cfgs['11']['normal_dwell']);normal.fit(caches['fit']);normal.calibrate(caches['calibration'])
    fitted={};cal={n:load(roots[n]/'normal_calibration_scores.npz') for n in ns}
    for n in ns[1:]:
        cfg=cfgs[n];record=json.loads(Path(f'results/experiment{n}/pre_evaluation_protocol.json').read_text())
        assert hashlib.sha256(Path(f'configs/experiment{n}.json').read_bytes()).hexdigest()==record['config_sha256']
        for path,sha in record['code_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
        for name,sha in record['source_sha256'].items():
            data=(roots[n]/'features'/scene/name).read_bytes();assert hashlib.sha256(data).hexdigest()==sha
            assert data==(roots['11']/'features'/scene/name).read_bytes()
        for key in base:
            if key not in ['threshold','dwell_reference']:np.testing.assert_array_equal(base[key],models[n][key],err_msg=n+':'+key)
        m=ContextDwell(**cfg['normal_dwell'],**cfg['dwell_context']);m.fit(caches['fit']);m.calibrate(caches['calibration']);fitted[n]=m
        np.testing.assert_array_equal(m.reference,models[n]['dwell_reference'])
        for (a,b),values in m.context_durations.items():np.testing.assert_array_equal(values,models[n][f'dwell_context_durations_{a}_{b}'])
        for key in ['visual','transition','dwell_age','phases','sequence']:np.testing.assert_array_equal(cal['11'][key],cal[n][key])
        for key in ['dwell','dwell_valid','dwell_age','dwell_reason']:
            field={'dwell':0,'dwell_valid':1,'dwell_age':2,'dwell_reason':3}[key]
            expected=np.concatenate([m.score(d)[field] for d in caches['calibration']]);np.testing.assert_array_equal(cal[n][key],expected)
        assert np.all(~cal[n]['dwell_valid']|cal['11']['dwell_valid'])
        np.testing.assert_array_equal(cal[n]['process'],np.where(cal[n]['dwell_valid'],np.maximum(cal[n]['transition'],cal[n]['dwell']),cal[n]['transition']))
        np.testing.assert_array_equal(cal[n]['combined'],np.maximum(cal[n]['visual'],cal[n]['process']))
        assert float(models[n]['threshold'])==np.quantile(cal[n]['combined'],cfg['calibration_quantile'],method='higher')
    np.testing.assert_array_equal(cal['12']['dwell_valid'],cal['12_support']['dwell_valid'])
    # The control retains the original state-only raw model on identical supported inputs.
    for d in caches['fit']+caches['calibration']:
        raw,valid,_,_=fitted['12_support'].raw(d);np.testing.assert_array_equal(raw[valid],normal.raw(d)[0][valid])
        np.testing.assert_array_equal(valid,fitted['12'].raw(d)[1])
    gathered={n:[] for n in ns};names=sorted(p.name for p in (roots['11']/'predictions').glob('*.npz'))
    for n in ns[1:]:assert names==sorted(p.name for p in (roots[n]/'predictions').glob('*.npz'))
    contexts=[];phases=[]
    for name in names:
        seq=Path(name).stem.split('_')[1];data=load(roots['11']/'features'/scene/f'testing_{seq}.npz');pred={n:load(roots[n]/'predictions'/name) for n in ns};old=pred['11']
        for n in ns[1:]:
            d=pred[n]
            for key in old:
                if key not in ['combined','process','dwell','dwell_valid','dwell_reason']:np.testing.assert_array_equal(old[key],d[key],err_msg=n+':'+name+':'+key)
            values=fitted[n].score(data)
            for i,key in enumerate(['dwell','dwell_valid','dwell_age','dwell_reason']):np.testing.assert_array_equal(d[key],hold_scores(data['indices'],values[i],len(d['labels'])))
            np.testing.assert_array_equal(d['dwell_entry_context'],hold_scores(data['indices'],observed_entry_context(data),len(d['labels'])))
            assert np.all(~d['dwell_valid']|old['dwell_valid'])
            np.testing.assert_array_equal(d['process'],np.where(d['dwell_valid'],np.maximum(d['transition'],d['dwell']),d['transition']))
            np.testing.assert_array_equal(d['combined'],np.maximum(d['visual'],d['process']))
            assert np.isfinite(d['combined']).all()
        np.testing.assert_array_equal(pred['12']['dwell_valid'],pred['12_support']['dwell_valid'])
        for n in ns:gathered[n].append(pred[n])
        contexts.append(pred['12']['dwell_entry_context']);phases.append(hold_scores(data['indices'],data['phases'],len(old['labels'])))
    joined={n:{k:np.concatenate([p[k] for p in gathered[n]]) for k in ['labels','combined','dwell_valid','dwell']} for n in ns}
    y=joined['11']['labels'];context=np.concatenate(contexts);phase=np.concatenate(phases)
    def counts(mask):return {'normal':int((mask&(y==0)).sum()),'anomaly':int((mask&(y==1)).sum())}
    alarms={n:joined[n]['combined']>float(models[n]['threshold']) for n in ns};pairs={}
    for before,after in [('11','12_support'),('12_support','12'),('11','12')]:
        pairs[f'{before}->{after}']={'added':counts(alarms[after]&~alarms[before]),'lost':counts(alarms[before]&~alarms[after]),'both':counts(alarms[after]&alarms[before])}
    by_context={}
    for a,b in np.unique(np.column_stack([context,phase]),axis=0):
        use=(context==a)&(phase==b)&joined['12']['dwell_valid']
        if use.any():by_context[f'{a}->{b}']={'frames':counts(use),'control_alarms':counts(use&alarms['12_support']),'entry_alarms':counts(use&alarms['12']),
            'entry_added_vs_control':counts(use&alarms['12']&~alarms['12_support']),'entry_lost_vs_control':counts(use&~alarms['12']&alarms['12_support'])}
    result={'scene':scene,'thresholds':{n:float(models[n]['threshold']) for n in ns},'valid_test_frames':{n:counts(joined[n]['dwell_valid']) for n in ns},'alarm_changes':pairs,'supported_context_strata':by_context,
        'dwell_saturated_at_one':{n:counts(joined[n]['dwell_valid']&(joined[n]['dwell']==1)) for n in ns},
        'note':'All thresholds fit on normal calibration per model. Support control changes gate and its resulting calibration; entry versus control isolates duration conditioning and its calibration on exactly matched observations.'}
    out=Path('results/comparison11_12_support_12');out.mkdir(parents=True,exist_ok=True);(out/'context_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    validation={'feature_sequences_checked_per_arm':len(record['source_sha256']),'test_predictions_checked_per_arm':len(names),'same_features_phases_visual_transition_objects_labels':True,
        'same_normal_state_models_except_dwell_reference_and_threshold':True,'context_reference_and_scores_reconstructed':True,'identical_context_masks_in_both_arms':True,
        'control_raw_equals_original_on_supported_observations':True,'normal_thresholds_reconstructed':True,'config_code_and_source_hashes_match_freeze':True,'artifact_validation_only':True}
    for n in ns[1:]:Path(f'results/experiment{n}/validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
