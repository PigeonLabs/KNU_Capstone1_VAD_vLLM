"""Experiment26 reconstruction of both full-normal references, causal gates and every score."""
import copy,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from ipad_vad.scoring import observations
from ipad_vad.data import hold_scores,evaluation_labels
from ipad_vad.events import anomaly_events,summarize_events
from evaluate_route_holdout import load_cache,normal_model_arrays
from audit_missing_age import STATE_NAMES,PROCESS,sha,load,direct_routes as legacy_routes,verify_calibration
from audit_bank_dispatch import VARIANTS,BEFORE,AFTER,PAIRS,verify_references,same_bank_checks

def direct_routes(data,variant,tau):return legacy_routes(data,{'hold':'23_obs','pool':'24_pool','age':'24_age'}[variant.split('_')[-1]],tau)
from audit_factorial_normal import bank_usage
from validate_factorial_appearance import metric


def main():
    out=Path('results/experiment26');art=Path('artifacts/experiment26');root=art/'features/R04';split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];protocol=json.loads((out/'pre_normal_protocol.json').read_text());profile=json.loads(Path('results/experiment24/fit_gap_profile.json').read_text());tau=profile['tau_frames']
    for record in [protocol,json.loads((out/'pre_test_checkpoint.json').read_text())]:
        for p,h in record['file_sha256'].items():assert sha(p)==h,p
    assert len(protocol['source_features_sha256'])==44
    for p,h in protocol['source_features_sha256'].items():
        for e in ['26',*AFTER]:assert sha(Path(f'artifacts/experiment{e}/features/R04')/p)==h
    fit=[load_cache(root,s) for s in split['fit']];cal={s:load_cache(root,s) for s in split['calibration']};audit=json.loads((out/'normal_audit.json').read_text());holdout=json.loads((out/'normal_holdout.json').read_text());models={};metrics={e:json.loads(Path(f'results/experiment{e}/metrics.json').read_text()) for e in VARIANTS};events={e:[] for e in VARIANTS};scores={e:{k:[] for k in ['visual','process','combined']} for e in VARIANTS};y=[];states=[];usage={e:{} for e in VARIANTS};test_routes={e:{s:{'global_phase_samples':0,'global_pooled_samples':0} for s in STATE_NAMES} for e in VARIANTS}
    # Independently reconstruct every bounded/unbounded normal gap, including censored metadata.
    complete=[];censored=[]
    for d in fit:
        i=0;v=d['relation_valid'];idx=d['indices']
        while i<len(v):
            if v[i]:i+=1;continue
            start=i
            while i<len(v) and not v[i]:i+=1
            row={'sequence_id':d['sequence_id'],'start_sample':start,'end_sample_exclusive':i,'samples':i-start,'start_frame':int(idx[start]),'last_missing_frame':int(idx[i-1])}
            if start>0 and i<len(v):row['duration_frames']=int(idx[i]-idx[start]);complete.append(row)
            else:row.update(left_censored=start==0,right_censored=i==len(v),observed_span_lower_bound_frames=int(idx[i-1]-idx[start]));censored.append(row)
    assert complete==profile['complete_runs'] and censored==profile['censored_runs'];assert tau==int(np.quantile([r['duration_frames'] for r in complete],.9,method='higher'))
    with threadpool_limits(limits=4):
        for e in VARIANTS:
            cfg=json.loads(Path(f'configs/experiment{e}.json').read_text());m=LognormalDwellBaseline(cfg,load_process(cfg));m.fit(fit);models[e]=m;saved=load(f'artifacts/experiment{e}/normal_model.npz');assert set(m.spaces)==set(models['25_hold'].spaces)
            for k,s in m.spaces.items():
                assert s.n==models['25_hold'].spaces[k].n
                for name in ['mean','basis']:
                    np.testing.assert_array_equal(getattr(s,name),saved[f'{name}_{k[0]}_{k[1]}']);np.testing.assert_array_equal(getattr(s,name),getattr(models['25_hold'].spaces[k],name))
            np.testing.assert_array_equal(m.transition,models['25_hold'].transition)
            if e in ['25_age','26_age']:
                assert m.missing_age.report()==profile==metrics[e]['appearance_missing_age'];assert saved['appearance_age_tau']==tau;np.testing.assert_array_equal(saved['appearance_age_fit_durations'],m.missing_age.durations)
        assert len(holdout['folds'])==30;fold_process={};fold_request_refs={}
        for row in holdout['folds']:
            e=row['variant'];held=row['held_out_sequence'];used=[s for s in split['calibration'] if s!=held];assert row['calibration_sequences']==used and held not in used;m=copy.deepcopy(models[e]);m.calibrate([cal[s] for s in used])
            if e in VARIANTS:
                verify_references(m,[cal[s] for s in used])
                for key,v in m.request_calibration.references.items():
                    if (held,key) in fold_request_refs:np.testing.assert_array_equal(v,fold_request_refs[held,key])
                    else:fold_request_refs[held,key]=v
                for v in row['request_support']:assert set(v['videos'])=={f'R04/training_{s}' for s in used}
            else:raise AssertionError(e)
            refs=normal_model_arrays(m);saved=load(art/'normal_holdout'/f'{e}_exclude_{held}_references.npz');assert refs.keys()==saved.keys()
            for k,v in refs.items():np.testing.assert_array_equal(v,saved[k])
            if e in ['25_age','26_age']:assert m.missing_age.tau==tau and m.missing_age.report()==profile
            d=cal[held];r=m.score(d);pred=load(art/'normal_holdout'/f'{e}_exclude_{held}_scores.npz')
            for k in ['visual','combined',*PROCESS]:np.testing.assert_array_equal(r[k],pred[k])
            np.testing.assert_array_equal(np.concatenate([v for _,_,v in r['objects']]),pred['object_scores'])
            if held in fold_process:
                for k in PROCESS:np.testing.assert_array_equal(r[k],fold_process[held][k])
            else:fold_process[held]=r
            assert m.threshold==row['normal_q99'];alarm=r['combined']>m.threshold;dense=hold_scores(d['indices'],alarm,int(d['frame_count'])).astype(bool);valid=hold_scores(d['indices'],d['relation_valid'],len(dense)).astype(bool)
            assert int(alarm.sum())==row['held_out_sample_alarms'] and len(alarm)==row['held_out_samples'];assert int(dense.sum())==row['held_out_frame_alarms'] and len(dense)==row['held_out_frames']
            for v in [False,True]:assert row['relation_strata'][str(v)]=={'frames':int(np.sum(valid==v)),'alarms':int(np.sum(dense&(valid==v)))}
        for e,m in models.items():
            rows=[r for r in holdout['folds'] if r['variant']==e]
            for k in ['held_out_samples','held_out_sample_alarms','held_out_frames','held_out_frame_alarms','held_out_at_one_samples']:assert sum(r[k] for r in rows)==holdout['totals'][e][k]
            m.calibrate(list(cal.values()))
            verify_references(m,list(cal.values()))
            saved=load(f'artifacts/experiment{e}/normal_calibration_scores.npz');pre=load(art/'full_normal'/f'{e}_scores.npz')
            for k,v in pre.items():np.testing.assert_array_equal(saved[k],v)
            for k in ['visual','combined',*PROCESS]:np.testing.assert_array_equal(np.concatenate([m.score(d)[k] for d in cal.values()]),saved[k])
            assert m.threshold==metrics[e]['normal_q99_threshold']==audit['variants'][e]['normal_q99']
            if e in VARIANTS:
                saved_model=load(f'artifacts/experiment{e}/normal_model.npz')
                for (role,request),v in m.request_calibration.references.items():
                    np.testing.assert_array_equal(v,saved_model[f'request_calibration_{role}_{request}']);np.testing.assert_array_equal(v,models['25_hold'].request_calibration.references[role,request])
        for p in sorted(root.glob('*.npz')):
            part,seq=p.stem.split('_');d=load_cache(root,seq,part);group='test' if part=='testing' else 'fit' if seq in split['fit'] else 'calibration';actual_age,actual_state,actual_phase=models['26_age'].missing_age.states(d);ph,st,ag=direct_routes(d,'26_age',tau)
            for actual,expected in [(actual_age,ag),(actual_state,st),(actual_phase,ph)]:np.testing.assert_array_equal(actual,expected)
            for length in sorted(set([1,len(ag)//2,len(ag)])):
                prefix={k:d[k][:length] for k in ['indices','phases','relation_valid']}
                for actual,expected in zip(models['26_age'].missing_age.states(prefix),(ag,st,ph)):np.testing.assert_array_equal(actual,expected[:length])
            base_raw=models['25_hold'].raw(d)[0];pool_raw=models['25_pool'].raw(d)[0]
            for e,m in models.items():
                phases,_,_=direct_routes(d,e,tau);np.testing.assert_array_equal(phases,m.appearance_phases(d));raw=m.raw(d)[0]
                for (role,frames,x),(r,f,v),(_,_,base),(_,_,pool) in zip(observations(d),raw,base_raw,pool_raw):
                    assert role==r;np.testing.assert_array_equal(frames,f);expected=np.empty(len(x))
                    for phase in np.unique(phases[frames]):
                        key=(role,int(phase))
                        if key not in m.spaces:key=(role,-1)
                        if key not in m.spaces:key=(-1,-1)
                        mask=phases[frames]==phase;expected[mask]=m.spaces[key].residual(x[mask])
                    np.testing.assert_allclose(v,expected,rtol=1e-12,atol=1e-12);obs=st[frames]==0;np.testing.assert_allclose(v[obs],base[obs],rtol=1e-12,atol=1e-12)
                    if e in ['25_age','26_age']:
                        young=st[frames]==2;fallback=np.isin(st[frames],[1,3]);np.testing.assert_allclose(v[young],base[young],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(v[fallback],pool[fallback],rtol=1e-12,atol=1e-12)
                counts=usage[e].setdefault(group,{})
                for k,v in bank_usage(m,d).items():counts[k]=counts.get(k,0)+v
                if part=='testing':
                    routes=m.appearance_routes(d,-1,np.arange(len(st)))
                    for i,name in enumerate(STATE_NAMES):
                        test_routes[e][name]['global_phase_samples']+=int(np.sum((st==i)&(routes==1)));test_routes[e][name]['global_pooled_samples']+=int(np.sum((st==i)&(routes==0)))
        support_fallback={e:{s:{'feature_observations':0,'changed_scores':0} for s in STATE_NAMES} for e in AFTER}
        for p in sorted(root.glob('*.npz')):
            part,seq=p.stem.split('_');d=load_cache(root,seq,part);st=direct_routes(d,'26_age',tau)[1]
            for e in AFTER:
                new,old=models[e],models[PAIRS[e]];ph=new.appearance_phases(d)
                for (role,f,v),(_,_,b),(_,_,raw),(_,_,oldraw) in zip(new.score(d)['objects'],old.score(d)['objects'],new.raw(d)[0],old.raw(d)[0]):
                    fallback=np.array([ph[i]>=0 and new.appearance_space_key(role,ph[i])[1]<0 for i in f]);np.testing.assert_array_equal(v[~fallback],b[~fallback]);np.testing.assert_allclose(raw,oldraw,rtol=1e-12,atol=1e-12)
                    if part=='testing':
                        for i,name in enumerate(STATE_NAMES):
                            mask=fallback&(st[f]==i);support_fallback[e][name]['feature_observations']+=int(mask.sum());support_fallback[e][name]['changed_scores']+=int(np.sum(mask&(v!=b)))
        gate_invariance=same_bank_checks(models,[load_cache(root,p.stem.split('_')[1],p.stem.split('_')[0]) for p in sorted(root.glob('*.npz'))])
        same_bank_changed_request={}
        for left,right in [('25_hold','25_pool'),('25_hold','25_age'),('25_pool','25_age'),('26_hold','26_pool'),('26_hold','26_age'),('26_pool','26_age')]:
            count=changed=0
            for p in sorted(root.glob('testing_*.npz')):
                data=load_cache(root,p.stem.split('_')[1],'testing');lp=models[left].appearance_phases(data);rp=models[right].appearance_phases(data)
                for (role,frames,lv),(_,_,rv) in zip(models[left].score(data)['objects'],models[right].score(data)['objects']):
                    mask=np.array([(lp[f]>=0)!=(rp[f]>=0) and models[left].appearance_space_key(role,lp[f])==models[right].appearance_space_key(role,rp[f]) for f in frames]);count+=int(mask.sum());changed+=int(np.sum(mask&(lv!=rv)))
            if left.startswith('26'):assert changed==0
            same_bank_changed_request[f'{left}_to_{right}']={'feature_observations':count,'changed_calibrated_scores':changed}
        for p in sorted(root.glob('testing_*.npz')):
            seq=p.stem.split('_')[1];d=load_cache(root,seq,'testing');n=int(d['frame_count']);labels=evaluation_labels(np.load(Path('/media/jeong/ExtHDD/IPAD_vLLM/IPAD_dataset/R04/test_label')/f'{int(seq):03}.npy'),n);y.append(labels);states.append(hold_scores(d['indices'],direct_routes(d,'26_age',tau)[1],n));reference=None
            for e,m in models.items():
                saved=load(f'artifacts/experiment{e}/predictions/R04_{seq}.npz');r=m.score(d);np.testing.assert_array_equal(labels,saved['labels'])
                for k in ['visual','combined',*PROCESS]:np.testing.assert_array_equal(hold_scores(d['indices'],r[k],n),saved[k])
                np.testing.assert_array_equal(np.concatenate([v for _,_,v in r['objects']]),saved['object_scores'])
                if reference is None:reference=saved
                else:
                    for k in ['labels','indices','phases','boxes','tracks','detected_roles','detected_object_frames',*PROCESS]:np.testing.assert_array_equal(saved[k],reference[k])
                for k in scores[e]:scores[e][k].append(saved[k])
                events[e].extend([dict(v,sequence_key=f'R04_{seq}') for v in anomaly_events(labels,saved['combined']>m.threshold)])
    y=np.concatenate(y);states=np.concatenate(states);assert len(y)==8154;scores={e:{k:np.concatenate(v) for k,v in s.items()} for e,s in scores.items()};alarms={e:s['combined']>models[e].threshold for e,s in scores.items()}
    def counts(mask):return {'normal':int(np.sum(mask&(y==0))),'anomaly':int(np.sum(mask&(y==1)))}
    summary={};strata={};pairs={}
    for e in VARIANTS:
        for k,s in scores[e].items():assert metric(y,s)==metrics[e]['metrics'][k]
        assert float(np.mean(alarms[e][y==0]))==metrics[e]['test_normal_frame_alarm_rate'];assert float(np.mean(alarms[e][y==1]))==metrics[e]['test_anomaly_frame_recall_at_q99']
        summary[e]={'q99':models[e].threshold,'alarms':counts(alarms[e]),'normal_fpr':float(np.mean(alarms[e][y==0])),'anomaly_recall':float(np.mean(alarms[e][y==1])),'metrics':metrics[e]['metrics'],'events':summarize_events(events[e]),'normal_holdout_fp':holdout['totals'][e]['held_out_frame_alarms']};strata[e]={}
        for i,name in enumerate(STATE_NAMES):
            mask=states==i;strata[e][name]={'frames':int(mask.sum()),'normal_frames':int(np.sum(mask&(y==0))),'anomaly_frames':int(np.sum(mask&(y==1))),'alarms':counts(mask&alarms[e]),'metrics':{k:metric(y[mask],s[mask]) for k,s in scores[e].items()},'bank_usage':test_routes[e][name]}
    for before,after in [('25_hold','26_hold'),('25_pool','26_pool'),('25_age','26_age'),('26_hold','26_pool'),('26_hold','26_age'),('26_pool','26_age')]:
        name=f'{before}_to_{after}';added=alarms[after]&~alarms[before];removed=alarms[before]&~alarms[after];pairs[name]={'before':before,'after':after,'added':counts(added),'removed':counts(removed),'age_strata':{s:{'added':counts(added&(states==i)),'removed':counts(removed&(states==i)),'combined_changed_frames':int(np.sum((states==i)&(scores[before]['combined']!=scores[after]['combined'])))} for i,s in enumerate(STATE_NAMES)},'gained_events':[],'lost_events':[]}
        for a,b in zip(events[before],events[after]):
            identity={k:a[k] for k in ['sequence_key','start_frame','end_frame_exclusive']};assert identity=={k:b[k] for k in identity}
            if b['detected'] and not a['detected']:pairs[name]['gained_events'].append(identity)
            if a['detected'] and not b['detected']:pairs[name]['lost_events'].append(identity)
    for left,right in [('26_hold','26_pool'),('26_hold','26_age'),('26_pool','26_age')]:
        for k in ['visual','combined','process']:np.testing.assert_array_equal(scores[left][k][np.isin(states,[0,1])],scores[right][k][np.isin(states,[0,1])])
    (out/'bank_dispatch_diagnostic.json').write_text(json.dumps({'tau_frames':tau,'variants':summary,'age_strata':strata,'paired_contrasts':pairs,'bank_usage_observations':usage,'same_actual_bank_feature_score_comparisons':gate_invariance,'same_bank_different_request':same_bank_changed_request,'support_fallback_changes':support_fallback,'events':events,'note':'Both full-normal reference arrays preserved from experiment25; new CDF dispatch uses actual subspace family, with own normal q99. Same raw PCA residuals within 1e-12; canonical batching stabilizes numeric ties. Repeated R04 development evaluation, no point adjustment or threshold sweep.'},indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps({'feature_sequences_checked':44,'normal_holdout_configurations_checked':30,'test_predictions_checked':114,'protocol_hashes_match':True,'feature_bytes_unchanged':True,'complete_censored_gap_metadata_independently_reconstructed':True,'tau_fit_only_frozen_and_same_all_folds':True,'all_banks_and_process_preserved':True,'causal_routes_and_prefix_invariance_verified':True,'observed_raw_unchanged_age_young_equals_hold_old_equals_pool':True,'normal_holdout_exclusion_references_q99_reconstructed':True,'all_legacy_variants_reproduced':True,'both_full_normal_references_independently_reconstructed':True,'same_actual_bank_calibrated_scores_identical_across_gates':True,'normal_preflight_matches_final':True,'scores_source_labels_and_metrics_recomputed':True},indent=2)+'\n');print(json.dumps({'summary':summary,'contrasts':pairs},indent=2))


if __name__=='__main__':main()
