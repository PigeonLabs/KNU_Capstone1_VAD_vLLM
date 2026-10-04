"""Verify experiment16 frozen inputs, unchanged branches and continuous duration scores."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import lognorm,norm
from sklearn.metrics import roc_auc_score,average_precision_score
from ipad_vad.data import hold_scores


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    out=Path('results/experiment16');cfg=json.loads(Path('configs/experiment16.json').read_text());freeze=json.loads((out/'pre_evaluation_protocol.json').read_text())
    for name,sha in freeze['file_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    root=Path('artifacts/experiment16');oldroot=Path('artifacts/experiment15')
    for name,sha in freeze['feature_sha256'].items():
        p=root/'features/R04'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha and p.read_bytes()==(oldroot/'features/R04'/name).read_bytes()
    old=load(oldroot/'normal_model.npz');new=load(root/'normal_model.npz')
    for key in old:
        if key!='threshold':np.testing.assert_array_equal(old[key],new[key])
    parameters={}
    for key,values in new.items():
        if key.startswith('dwell_log_parameters_'):
            a,b=map(int,key.split('_')[-2:]);durations=new[f'dwell_context_durations_{a}_{b}'];logs=np.log(durations)
            np.testing.assert_array_equal(values,[logs.mean(),max(logs.std(ddof=0),cfg['dwell_lognormal']['sigma_floor'])]);parameters[(a,b)]=values
    cal=load(root/'normal_calibration_scores.npz');preflight=load(root/'preflight_calibration_scores.npz');oldcal=load(oldroot/'normal_calibration_scores.npz')
    for key in preflight:np.testing.assert_array_equal(preflight[key],cal[key])
    for key in oldcal:
        if key not in ['process','combined','dwell']:np.testing.assert_array_equal(oldcal[key],cal[key])
    assert float(new['threshold'])==np.quantile(cal['combined'],.99,method='higher')
    def reconstruct(d,phase):
        expected=np.zeros(len(phase))
        for (a,b),(mu,sigma) in parameters.items():
            use=d['dwell_valid']&(d['dwell_entry_context']==a)&(phase==b)&(d['dwell_age']>0)
            expected[use]=lognorm.cdf(d['dwell_age'][use],s=sigma,scale=np.exp(mu))
        np.testing.assert_allclose(expected,d['dwell'],rtol=0,atol=1e-14)
        np.testing.assert_array_equal(d['process'],np.where(d['dwell_valid'],np.maximum(d['transition'],d['dwell']),d['transition']))
        np.testing.assert_array_equal(d['combined'],np.maximum(d['visual'],d['process']))
    reconstruct(cal,cal['phases'])
    arrays={key:[] for key in ['labels','visual','process','combined','dwell','dwell_valid','dwell_age']};before15=[];beforebase=[];phases=[];contexts=[]
    baseq=json.loads(Path('results/experiment15_base/metrics.json').read_text())['normal_q99_threshold']
    paths=sorted((root/'predictions').glob('*.npz'))
    for path in paths:
        a=load(oldroot/'predictions'/path.name);b=load(path);base=load(Path('artifacts/experiment15_base/predictions')/path.name)
        for key in a:
            if key not in ['process','combined','dwell']:np.testing.assert_array_equal(a[key],b[key])
        phase=hold_scores(b['indices'],b['phases'],len(b['labels']));reconstruct(b,phase)
        for key in arrays:arrays[key].append(b[key])
        before15.append(a['combined']>float(old['threshold']));beforebase.append(base['combined']>baseq);phases.append(phase);contexts.append(b['dwell_entry_context'])
    arrays={k:np.concatenate(v) for k,v in arrays.items()};y=arrays['labels'];after=arrays['combined']>float(new['threshold']);before15=np.concatenate(before15);beforebase=np.concatenate(beforebase);phases=np.concatenate(phases);contexts=np.concatenate(contexts)
    m=json.loads((out/'metrics.json').read_text());assert len(y)==8154 and len(paths)==19
    for key in ['visual','process','combined']:
        assert np.isfinite(arrays[key]).all()
        assert m['metrics'][key]['auroc']==roc_auc_score(y,arrays[key]) and m['metrics'][key]['average_precision']==average_precision_score(y,arrays[key])
    assert m['test_normal_frame_alarm_rate']==np.mean(after[y==0]) and m['test_anomaly_frame_recall_at_q99']==np.mean(after[y==1])
    def counts(mask):return {'normal':int(np.sum(mask&(y==0))),'anomaly':int(np.sum(mask&(y==1)))}
    comparisons={n:{'added':counts(after&~before),'removed':counts(before&~after)} for n,before in [('15',before15),('15_base',beforebase)]}
    strata={}
    for a,b in parameters:
        use=arrays['dwell_valid']&(contexts==a)&(phases==b);strata[f'{a}->{b}']={'valid_frames':counts(use),'added_over_no_dwell':counts(use&after&~beforebase)}
    tail={}
    for (a,b),(mu,sigma) in parameters.items():
        use=arrays['dwell_valid']&(contexts==a)&(phases==b)
        tail[f'{a}->{b}']={'max_age':float(arrays['dwell_age'][use].max()) if use.any() else None,'max_score':float(arrays['dwell'][use].max()) if use.any() else None,'exceed_q99':int(np.sum(use&(arrays['dwell']>float(new['threshold'])))),'age_at_q99':float(np.exp(mu+sigma*norm.ppf(float(new['threshold']))))}
    (out/'tail_operating_range.json').write_text(json.dumps({'q99':float(new['threshold']),'contexts':tail,'note':'Age at q99 is an analytic diagnostic, not a selected decision threshold. Source-frame units; no FPS assumed.'},indent=2)+'\n')
    result={'comparisons':comparisons,'q99_before':float(old['threshold']),'q99_after':float(new['threshold']),'q99_no_dwell':baseq,'test_dwell_at_one':counts(arrays['dwell_valid']&(arrays['dwell']==1)),'test_combined_at_one':counts(arrays['combined']==1),'added_contexts':strata,'note':'Shared features/support; each normal q99. Development R04, inherited grounding errors unchanged.'}
    (out/'lognormal_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps({'feature_sequences_checked':len(freeze['feature_sha256']),'test_sequences_checked':len(paths),'frozen_input_and_code_hashes_match':True,'same_features_duration_banks_visual_transition_masks_ages_and_labels':True,'parameters_and_scores_reconstructed':True,'cdf_reconstruction_absolute_tolerance':1e-14,'normal_preflight_scores_match_final_calibration':True,'normal_q99_and_metrics_recomputed':True},indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
