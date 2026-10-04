"""Validate experiment19 provenance, routing, preserved process and full metrics."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.scoring import Subspace,observations
from ipad_vad.experiment import load_process
from ipad_vad.data import hold_scores,evaluation_labels
from prepare_observed_appearance import usage


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def metric(y,s):
    return {'frames':len(y),'positive_frames':int(y.sum()),'auroc':float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,'average_precision':float(average_precision_score(y,s)) if np.any(y==1) else None}


def main():
    out=Path('results/experiment19');root=Path('artifacts/experiment19');cfg=json.loads(Path('configs/experiment19.json').read_text());split=json.loads(Path('results/stage00/splits.json').read_text())['R04']
    source=Path('artifacts/experiment18/features/R04');target=root/'features/R04'
    protocol=json.loads((out/'pre_normal_protocol.json').read_text());checkpoint=json.loads((out/'pre_test_checkpoint.json').read_text())
    for record in [protocol,checkpoint]:
        for name,h in record['file_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    paths=sorted(source.glob('*.npz'));assert len(paths)==44
    for p in paths:
        expected=protocol['source_features_sha256'][p.name]
        assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256((target/p.name).read_bytes()).hexdigest()==expected
    fit=[load(target/f'training_{s}.npz') for s in split['fit']];cal=[load(target/f'training_{s}.npz') for s in split['calibration']]
    old_cfg=json.loads(Path('configs/experiment18.json').read_text());old_saved=load('artifacts/experiment18/normal_model.npz');saved=load(root/'normal_model.npz')
    m=json.loads((out/'metrics.json').read_text());old_m=json.loads(Path('results/experiment18/metrics.json').read_text())
    process_keys=['transition','dwell','process','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']
    cal_diagnostics=[];bank_usage={};label_parts=[];before=[];after=[];valid_parts=[];scores={e:{k:[] for k in ['visual','process','combined']} for e in ['18','19']}
    with threadpool_limits(limits=4):
        model=LognormalDwellBaseline(cfg,load_process(cfg));model.fit(fit);model.calibrate(cal)
        old=LognormalDwellBaseline(old_cfg,load_process(old_cfg));old.fit(fit);old.calibrate(cal)
        assert model.threshold==m['normal_q99_threshold']==float(saved['threshold'])
        # Independently reconstruct each bank from selected original rows.
        banks={}
        for d in fit:
            for role,frames,x in observations(d):
                banks.setdefault((role,-1),[]).append(x)
                for phase in range(model.k):
                    keep=d['relation_valid'][frames]&(d['phases'][frames]==phase)
                    if keep.any():banks.setdefault((role,phase),[]).append(x[keep])
        expected={k:np.concatenate(v) for k,v in banks.items() if sum(len(a) for a in v)>=cfg['minimum_phase_samples']}
        assert set(expected)==set(model.spaces)
        for (role,phase),x in expected.items():
            independent=Subspace(x,cfg['pca_variance'],cfg['pca_max_rank']);s=model.spaces[role,phase];assert s.n==len(x)
            np.testing.assert_array_equal(independent.mean,s.mean);np.testing.assert_array_equal(independent.basis,s.basis)
            np.testing.assert_array_equal(s.mean,saved[f'mean_{role}_{phase}']);np.testing.assert_array_equal(s.basis,saved[f'basis_{role}_{phase}'])
            if phase==-1:
                np.testing.assert_array_equal(s.mean,old_saved[f'mean_{role}_{phase}']);np.testing.assert_array_equal(s.basis,old_saved[f'basis_{role}_{phase}'])
        for k in old_saved:
            if k.startswith(('transition','process_reference','dwell_')):np.testing.assert_array_equal(old_saved[k],saved[k])
        cal_saved=load(root/'normal_calibration_scores.npz');pre=load(root/'preflight_calibration_scores.npz');cal_old=load('artifacts/experiment18/normal_calibration_scores.npz')
        for k in pre:np.testing.assert_array_equal(pre[k],cal_saved[k])
        for k in process_keys:np.testing.assert_array_equal(cal_saved[k],cal_old[k])
        # Check raw process invariance and every raw appearance route for all videos.
        for p in paths:
            d=load(target/p.name);obs,raw_process=model.raw(d);np.testing.assert_array_equal(raw_process,old.raw(d)[1])
            for (role,frames,x),(_,_,residual) in zip(observations(d),obs):
                for phase in np.unique(d['phases'][frames]):
                    for observed in [False,True]:
                        mask=(d['phases'][frames]==phase)&(d['relation_valid'][frames]==observed)
                        if not mask.any():continue
                        key=(role,int(phase)) if observed and (role,int(phase)) in model.spaces else (role,-1)
                        if key not in model.spaces:key=(-1,-1)
                        np.testing.assert_allclose(residual[mask],model.spaces[key].residual(x[mask]),rtol=1e-12,atol=1e-12)
            part,seq=p.stem.split('_');group='test' if part=='testing' else 'fit' if seq in split['fit'] else 'calibration'
            dest=bank_usage.setdefault(group,{})
            for k,v in usage(model,d).items():dest[k]=dest.get(k,0)+v
        cal_results=[model.score(d) for d in cal]
        for k in ['visual','combined',*process_keys]:np.testing.assert_array_equal(np.concatenate([r[k] for r in cal_results]),cal_saved[k])
        # Descriptive normal calibration mixture diagnostic, no refit or selection.
        role_rows={}
        for d,r in zip(cal,cal_results):
            for (role,frames,raw),(_,_,s) in zip(model.raw(d)[0],r['objects']):
                for observed in [False,True]:
                    keep=d['relation_valid'][frames]==observed;key=(role,observed)
                    row=role_rows.setdefault(key,{'raw':[],'score':[]});row['raw'].extend(raw[keep]);row['score'].extend(s[keep])
        for (role,observed),row in sorted(role_rows.items()):
            raw=np.array(row['raw']);s=np.array(row['score'])
            cal_diagnostics.append({'role':role,'relation_observed':observed,'observations':len(raw),'raw_quantiles_50_90_99':np.quantile(raw,[.5,.9,.99]).tolist() if len(raw) else [],'calibrated_quantiles_50_90_99':np.quantile(s,[.5,.9,.99]).tolist() if len(s) else [],'above_final_q99':int(np.sum(s>model.threshold))})
        pred_paths=sorted((root/'predictions').glob('*.npz'));assert len(pred_paths)==19
        for p in pred_paths:
            a=load(Path('artifacts/experiment18/predictions')/p.name);b=load(p);seq=p.stem.split('_')[1];d=load(target/f'testing_{seq}.npz');r=model.score(d);n=int(d['frame_count'])
            truth=evaluation_labels(np.load(Path('/media/jeong/ExtHDD/IPAD_vLLM/IPAD_dataset/R04/test_label')/f'{int(seq):03}.npy'),n)
            np.testing.assert_array_equal(truth,b['labels'])
            for k in ['labels','indices','phases','boxes','tracks','detected_roles','detected_object_frames','object_roles','object_sample_indices','object_detection_indices',*process_keys]:np.testing.assert_array_equal(a[k],b[k])
            for k in ['visual','combined',*process_keys]:np.testing.assert_array_equal(hold_scores(d['indices'],r[k],n),b[k])
            np.testing.assert_array_equal(np.concatenate([v for _,_,v in r['objects']]),b['object_scores'])
            # Legacy refactor retains the saved experiment18 outputs.
            old_r=old.score(d)
            for k in ['visual','combined',*process_keys]:np.testing.assert_array_equal(hold_scores(d['indices'],old_r[k],n),a[k])
            label_parts.append(b['labels']);valid_parts.append(hold_scores(d['indices'],d['relation_valid'],n).astype(bool));before.append(a['combined']>old.threshold);after.append(b['combined']>model.threshold)
            for e,pred in [('18',a),('19',b)]:
                for k in scores[e]:scores[e][k].append(pred[k])
    y=np.concatenate(label_parts);valid=np.concatenate(valid_parts);before=np.concatenate(before);after=np.concatenate(after);assert len(y)==8154
    scores={e:{k:np.concatenate(v) for k,v in branches.items()} for e,branches in scores.items()}
    for k in scores['19']:assert metric(y,scores['19'][k])==m['metrics'][k]
    assert np.mean(after[y==0])==m['test_normal_frame_alarm_rate'] and np.mean(after[y==1])==m['test_anomaly_frame_recall_at_q99']
    def counts(mask):return {'normal':int(np.sum(mask&(y==0))),'anomaly':int(np.sum(mask&(y==1)))}
    strata={}
    for observed in [False,True]:
        mask=valid==observed;strata[str(observed)]={'frames':int(mask.sum()),'normal_frames':int(np.sum(mask&(y==0))),'anomaly_frames':int(np.sum(mask&(y==1))),'alarms_before':counts(mask&before),'alarms_after':counts(mask&after),'added':counts(mask&after&~before),'removed':counts(mask&before&~after),'metrics':{e:{k:metric(y[mask],s[mask]) for k,s in branches.items()} for e,branches in scores.items()}}
    result={'q99_before':old_m['normal_q99_threshold'],'q99_after':model.threshold,'added_alarms':counts(after&~before),'removed_alarms':counts(before&~after),'strata_relation_observed':strata,'bank_usage_observations':bank_usage,'normal_calibration_role_observation_strata':cal_diagnostics,'note':'Fixed masks and identical process scores. Each model refits appearance calibration and q99. Counts of bank usage include global and each crop, not unique frames. Calibration strata are descriptive and not tuned.'}
    (out/'appearance_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps({'feature_sequences_checked':44,'test_sequences_checked':19,'protocol_hashes_match':True,'all_feature_arrays_byte_identical':True,'normal_banks_independently_reconstructed':True,'pooled_banks_exactly_preserved':True,'all_appearance_routes_checked':True,'raw_and_calibrated_process_unchanged':True,'legacy_experiment18_reproduced':True,'preflight_and_final_calibration_identical':True,'scores_and_source_labels_reproduced':True,'metrics_recomputed':True},indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['bank_usage_observations','normal_calibration_role_observation_strata']},indent=2))


if __name__=='__main__':main()
