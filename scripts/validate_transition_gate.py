"""Reconstruct experiment27 normal exclusion, test scores, causal gates and branch contribution."""
import copy,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.data import hold_scores,evaluation_labels
from ipad_vad.events import anomaly_events,summarize_events
from ipad_vad.dwell_scoring import observed_transition_mask
from evaluate_route_holdout import load_cache,normal_model_arrays
from audit_transition_gate import VARIANTS,BEFORE,AFTER,PAIRS,PRESERVED,EXTRA,KEYS,direct_mask,evidence_state,fitted_models,paired_checks,verify_references,load,sha
from audit_missing_age import STATE_NAMES,direct_routes
from validate_factorial_appearance import metric

EVIDENCE_NAMES=['first_sample','observed_pair','missing','reacquired']


def main():
    out=Path('results/experiment27');art=Path('artifacts/experiment27');root=art/'features/R04';split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];protocol=json.loads((out/'pre_normal_protocol.json').read_text())
    for record in [protocol,json.loads((out/'pre_test_checkpoint.json').read_text())]:
        for p,h in record['file_sha256'].items():assert sha(p)==h,p
        for p,h in record['source_features_sha256'].items():
            for e in ['27',*AFTER]:assert sha(Path(f'artifacts/experiment{e}/features/R04')/p)==h
    assert len(protocol['source_features_sha256'])==44
    fit=[load_cache(root,s) for s in split['fit']];cal={s:load_cache(root,s) for s in split['calibration']};hold=json.loads((out/'normal_holdout.json').read_text());audit=json.loads((out/'normal_audit.json').read_text());metrics={e:json.loads(Path(f'results/experiment{e}/metrics.json').read_text()) for e in VARIANTS};normal_strata=[]
    with threadpool_limits(limits=4):
        models=fitted_models(fit);assert len(hold['folds'])==30;foldmodels={}
        for row in hold['folds']:
            e=row['variant'];held=row['held_out_sequence'];used=[s for s in split['calibration'] if s!=held];assert row['calibration_sequences']==used;model=copy.deepcopy(models[e]);model.calibrate([cal[s] for s in used]);foldmodels.setdefault(held,{})[e]=model;verify_references(model,[cal[s] for s in used]);refs=normal_model_arrays(model);saved=load(art/'normal_holdout'/f'{e}_exclude_{held}_references.npz');assert refs.keys()==saved.keys()
            for k,v in refs.items():np.testing.assert_array_equal(v,saved[k])
            for support in row['request_support']:assert set(support['videos'])=={f'R04/training_{s}' for s in used}
            d=cal[held];r=model.score(d);pred=load(art/'normal_holdout'/f'{e}_exclude_{held}_scores.npz')
            for k in KEYS+(EXTRA if e in AFTER else []):np.testing.assert_array_equal(r[k],pred[k])
            np.testing.assert_array_equal(np.concatenate([s for _,_,s in r['objects']]),pred['object_scores'])
            assert row['normal_q99']==model.threshold;alarm=r['combined']>model.threshold;dense=hold_scores(d['indices'],alarm,int(d['frame_count'])).astype(bool);state=hold_scores(d['indices'],evidence_state(d),len(dense))
            assert int(alarm.sum())==row['held_out_sample_alarms'] and int(dense.sum())==row['held_out_frame_alarms']
            normal_strata.append({'variant':e,'held_out':held,'states':{name:{'frames':int(np.sum(state==i)),'alarms':int(np.sum((state==i)&dense))} for i,name in enumerate(EVIDENCE_NAMES)},'previous_state_reference_counts':{str(k):len(v) for k,v in model.state_process_references.items()}})
        for held,fold in foldmodels.items():paired_checks(fold,[cal[held]])
        for e,m in models.items():
            rows=[r for r in hold['folds'] if r['variant']==e]
            for key in ['held_out_samples','held_out_sample_alarms','held_out_frames','held_out_frame_alarms','held_out_at_one_samples']:assert sum(r[key] for r in rows)==hold['totals'][e][key]
            m.calibrate(list(cal.values()));verify_references(m,list(cal.values()));saved=load(f'artifacts/experiment{e}/normal_calibration_scores.npz');pre=load(art/'full_normal'/f'{e}_scores.npz')
            for k,v in pre.items():np.testing.assert_array_equal(v,saved[k])
            for k in KEYS+(EXTRA if e in AFTER else []):np.testing.assert_array_equal(np.concatenate([m.score(d)[k] for d in cal.values()]),saved[k])
            assert m.threshold==metrics[e]['normal_q99_threshold']==audit['variants'][e]['normal_q99']
            sm=load(f'artifacts/experiment{e}/normal_model.npz')
            for k,v in normal_model_arrays(m).items():np.testing.assert_array_equal(v,sm[k])
            for (a,b),v in m.dwell.log_parameters.items():np.testing.assert_array_equal(v,sm[f'dwell_log_parameters_{a}_{b}'])
        allcaches=[load_cache(root,p.stem.split('_')[1],p.stem.split('_')[0]) for p in sorted(root.glob('*.npz'))];paired_checks(models,allcaches)
        for d in allcaches:
            mask=direct_mask(d);np.testing.assert_array_equal(mask,observed_transition_mask(d))
            for n in sorted(set([0,1,len(mask)//2,len(mask)])):np.testing.assert_array_equal(observed_transition_mask({k:d[k][:n] for k in ['phases','relation_valid']}),mask[:n])
        scores={e:{k:[] for k in KEYS+(EXTRA if e in AFTER else [])} for e in VARIANTS};labels=[];states=[];ages=[];events={e:[] for e in VARIANTS};fixed_events={e:[] for e in AFTER};visual_events={e:[] for e in VARIANTS}
        for p in sorted(root.glob('testing_*.npz')):
            seq=p.stem.split('_')[1];d=load_cache(root,seq,'testing');n=int(d['frame_count']);y=evaluation_labels(np.load(Path('/media/jeong/ExtHDD/IPAD_vLLM/IPAD_dataset/R04/test_label')/f'{int(seq):03}.npy'),n);labels.append(y);states.append(hold_scores(d['indices'],evidence_state(d),n));ages.append(hold_scores(d['indices'],direct_routes(d,'24_age',56)[1],n))
            for e,m in models.items():
                r=m.score(d);v=load(f'artifacts/experiment{e}/predictions/R04_{seq}.npz');np.testing.assert_array_equal(y,v['labels'])
                for k in scores[e]:np.testing.assert_array_equal(hold_scores(d['indices'],r[k],n),v[k]);scores[e][k].append(v[k])
                np.testing.assert_array_equal(np.concatenate([s for _,_,s in r['objects']]),v['object_scores'])
                for key in ['indices','phases','boxes','tracks']:np.testing.assert_array_equal(d[key],v[key])
                events[e].extend([dict(x,sequence_key=f'R04_{seq}') for x in anomaly_events(y,v['combined']>m.threshold)])
                visual_events[e].extend([dict(x,sequence_key=f'R04_{seq}') for x in anomaly_events(y,v['visual']>m.threshold)])
                if e in AFTER:fixed_events[e].extend([dict(x,sequence_key=f'R04_{seq}') for x in anomaly_events(y,v['combined']>models[PAIRS[e]].threshold)])
    y=np.concatenate(labels);state=np.concatenate(states);age=np.concatenate(ages);assert len(y)==8154;scores={e:{k:np.concatenate(v) for k,v in s.items()} for e,s in scores.items()};alarms={e:s['combined']>models[e].threshold for e,s in scores.items()}
    def counts(mask):return {'normal':int(np.sum(mask&(y==0))),'anomaly':int(np.sum(mask&(y==1)))}
    def event_changes(a,b):
        gains=[];losses=[]
        for x,z in zip(a,b):
            key={k:x[k] for k in ['sequence_key','start_frame','end_frame_exclusive']};assert key=={k:z[k] for k in key}
            if z['detected'] and not x['detected']:gains.append(key)
            if x['detected'] and not z['detected']:losses.append(key)
        return {'gained_events':gains,'lost_events':losses}
    summary={};strata={};age_strata={};branches={};pairs={};thresholds={}
    for e in VARIANTS:
        m=models[e];s=scores[e];a=alarms[e]
        for k in ['visual','process','combined']:assert metric(y,s[k])==metrics[e]['metrics'][k]
        assert float(a[y==0].mean())==metrics[e]['test_normal_frame_alarm_rate'];assert float(a[y==1].mean())==metrics[e]['test_anomaly_frame_recall_at_q99']
        summary[e]={'q99':m.threshold,'metrics':metrics[e]['metrics'],'alarms':counts(a),'normal_fpr':float(a[y==0].mean()),'anomaly_recall':float(a[y==1].mean()),'events':summarize_events(events[e]),'normal_holdout_fp':hold['totals'][e]['held_out_frame_alarms']}
        for target,vector,names in [(strata,state,EVIDENCE_NAMES),(age_strata,age,STATE_NAMES)]:
            target[e]={}
            for i,name in enumerate(names):
                mask=vector==i;target[e][name]={'frames':int(mask.sum()),**{k+'_frames':v for k,v in counts(mask).items()},'alarms':counts(a&mask),'metrics':{k:metric(y[mask],s[k][mask]) for k in ['visual','process','combined']}}
        transition=s['transition_gated'] if e in AFTER else s['transition'];dwell=np.where(s['dwell_valid'],s['dwell'],0);va=s['visual']>m.threshold;ta=transition>m.threshold;da=dwell>m.threshold
        np.testing.assert_array_equal(a,va|ta|da)
        branches[e]={'at_own_combined_normal_q99':m.threshold,'visual_alarms':counts(va),'transition_only_alarms':counts(ta&~va&~da),'dwell_only_alarms':counts(da&~va&~ta),'both_process_only_alarms':counts(ta&da&~va),'process_added_over_visual':counts(a&~va),'transition_strictly_dominates_visual_and_dwell':counts((transition>s['visual'])&(transition>dwell)),'dwell_strictly_dominates_visual_and_transition':counts((dwell>s['visual'])&(dwell>transition)),'visual_events_at_combined_q99':summarize_events(visual_events[e]),'process_event_contribution':event_changes(visual_events[e],events[e])}
    for e in AFTER:
        b=PAIRS[e];fixed=scores[e]['combined']>models[b].threshold;assert not np.any(fixed&~alarms[b]);assert models[e].threshold<=models[b].threshold
        pairs[f'{b}_to_{e}']={'added':counts(alarms[e]&~alarms[b]),'removed':counts(alarms[b]&~alarms[e]),'score_change_at_legacy_q99':{'added':counts(fixed&~alarms[b]),'removed':counts(alarms[b]&~fixed)},'states':{name:{'added':counts((state==i)&alarms[e]&~alarms[b]),'removed':counts((state==i)&alarms[b]&~alarms[e]),'combined_changed_frames':int(np.sum((state==i)&(scores[e]['combined']!=scores[b]['combined'])))} for i,name in enumerate(EVIDENCE_NAMES)},**event_changes(events[b],events[e])}
        thresholds[e]={'own_q99':models[e].threshold,'legacy_q99':models[b].threshold,'own_minus_legacy_on_same_new_scores':{'added':counts(alarms[e]&~fixed),'removed':counts(fixed&~alarms[e])},'fixed_q_events':summarize_events(fixed_events[e]),'own_q_events':summarize_events(events[e])}
    result={'variants':summary,'evidence_strata':strata,'age_strata':age_strata,'paired_contrasts':pairs,'threshold_decomposition':thresholds,'branch_contribution':branches,'normal_holdout_evidence_strata':normal_strata,'events':events,'note':'Repeated R04 development, paired fixed appearance/process models, own normal q99. First sampled state is causally held; subsets are dense frames, not independent observations. Visual-only contribution uses combined threshold, not separately calibrated visual q99.'}
    (out/'transition_gate_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');(out/'validation.json').write_text(json.dumps({'feature_files_checked':44,'normal_holdout_cells_checked':30,'test_predictions_checked':114,'protocol_and_feature_hashes_match':True,'all_paired_pca_references_appearance_ungated_transition_dwell_preserved':True,'normal_exclusion_q99_and_predictions_reconstructed':True,'causal_gate_boundaries_and_prefix_verified':True,'old_scores_reproduced':True,'new_process_combined_never_increase_at_fixed_q':True,'labels_metrics_branch_events_recomputed':True},indent=2)+'\n');print(json.dumps({'variants':summary,'pairs':pairs,'thresholds':thresholds,'branches':branches},indent=2))


if __name__=='__main__':main()
