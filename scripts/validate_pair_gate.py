"""Independent references, fixed components, branch contributions and threshold effects."""
import copy,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.data import hold_scores,evaluation_labels
from ipad_vad.events import anomaly_events,summarize_events
from ipad_vad.transition_evidence import same_track_pair_transition_mask
from evaluate_baseline import metrics
from evaluate_route_holdout import normal_model_arrays
from experiment31_pair_gate import (OUT,ART,ROOT,BEFORE,AFTER,VARIANTS,PAIRS,SPLIT,STATE_NAMES,KEYS,PRESERVED,load,load_cache,write,verify_freeze,verify_references,evidence,coverage,fitted_models,paired_checks,array_checks,model_arrays)


def process_check(m,caches):
    refs=[];states={i:[] for i in range(m.k)}
    for d in caches:
        p=d['phases'];raw=np.zeros(len(p));raw[1:]=-np.log(m.transition[p[:-1],p[1:]])+(~m.allowed[p[:-1],p[1:]])
        refs.extend(raw)
        for state in states:states[state].extend(raw[1:][p[:-1]==state])
    np.testing.assert_array_equal(refs,m.process_reference)
    for state,values in states.items():np.testing.assert_array_equal(values,m.state_process_references[state])
    for d in caches:
        r=m.score(d);expected=[];p=d['phases'];state=evidence(d)
        for i,x in enumerate(r['transition_raw']):
            ref=np.array(states[p[i-1]] if i and len(states[p[i-1]])>=m.minimum_support else refs);expected.append((np.sum(ref<x)+.5*np.sum(ref==x))/len(ref))
        np.testing.assert_array_equal(expected,r['transition']);gate=state==3 if m.cfg['transition_evidence_gate']=='same_track_pair' else state>=3
        np.testing.assert_array_equal(r['transition_gated'],np.where(gate,expected,0))


def causal_prefix(d):
    expected=evidence(d)==3
    for n in sorted(set([0,1,len(expected)//2,len(expected)])):
        ids=np.flatnonzero(d['object_frames']<n);remap=np.full(len(d['tracks']),-1,int);remap[ids]=np.arange(len(ids));selected=d['relation_detection_indices'][:n].copy();present=selected>=0;selected[present]=remap[selected[present]]
        prefix={'phases':d['phases'][:n],'relation_valid':d['relation_valid'][:n],'relation_detection_indices':selected,'tracks':d['tracks'][ids],'object_frames':d['object_frames'][ids]}
        np.testing.assert_array_equal(same_track_pair_transition_mask(prefix),expected[:n])


def main():
    verify_freeze('pre_normal_protocol.json');verify_freeze('pre_test_checkpoint.json');fit=[load_cache(ROOT,s) for s in SPLIT['fit']];cal={s:load_cache(ROOT,s) for s in SPLIT['calibration']};models=fitted_models(fit);audit=json.loads((OUT/'normal_audit.json').read_text());hold=json.loads((OUT/'normal_holdout.json').read_text());eligible=json.loads((OUT/'test_eligibility.json').read_text())['eligible'];assert eligible==AFTER
    normal_changes=[];counterexamples=[];fold_count=0
    for held,d in cal.items():
        used=[cal[s] for s in cal if s!=held];fold={};pred={}
        for e,base in models.items():
            m=copy.deepcopy(base);m.calibrate(used);verify_references(m,used);process_check(m,used);fold[e]=m;r=m.score(d);pred[e]=r;source=ART/'normal_holdout';array_checks(normal_model_arrays(m),load(source/f'{e}_exclude_{held}_references.npz'));saved=load(source/f'{e}_exclude_{held}_scores.npz');array_checks({k:r[k] for k in KEYS},saved);np.testing.assert_array_equal(np.concatenate([s for _,_,s in r['objects']]),saved['object_scores']);row=next(x for x in hold['folds'] if x['variant']==e and x['held_out_sequence']==held);assert row['calibration_sequences']==[s for s in cal if s!=held];assert row['normal_q99']==m.threshold
            for ref in row['request_support']:assert set(ref['videos'])=={x['sequence_id'] for x in used}
            dense=hold_scores(d['indices'],r['combined']>m.threshold,int(d['frame_count']));assert int(dense.sum())==row['held_out_frame_alarms'];fold_count+=1
            if e in BEFORE:array_checks({k:r[k] for k in KEYS},load(Path('artifacts/experiment30/normal_holdout')/f'{e}_exclude_{held}_scores.npz'))
            if held=='02':
                for frame in [76,252]:
                    i=int(np.flatnonzero(d['indices']==frame)[0]);counterexamples.append({'variant':e,'frame':frame,'evidence':STATE_NAMES[evidence(d)[i]],'q99':m.threshold,**{k:float(r[k][i]) for k in ['visual','transition','transition_gated','dwell','process','combined']}})
        paired_checks(fold,[d])
        for e in AFTER:
            old,new=pred[PAIRS[e]],pred[e];oa=old['combined']>fold[PAIRS[e]].threshold;na=new['combined']>fold[e].threshold;state=evidence(d);dense=lambda x:hold_scores(d['indices'],x,int(d['frame_count']))
            normal_changes.append({'variant':e,'held_out':held,'old_q99':fold[PAIRS[e]].threshold,'new_q99':fold[e].threshold,'removed_frames':int(dense(oa&~na).sum()),'added_frames':int(dense(na&~oa).sum()),'removed_sample_source_frames':d['indices'][oa&~na].tolist(),'added_sample_source_frames':d['indices'][na&~oa].tolist(),'closed_pair_boundary_frames':int(dense(state>=4).sum())})
    for e,m in models.items():
        m.calibrate(list(cal.values()));verify_references(m,list(cal.values()));process_check(m,list(cal.values()));array_checks(model_arrays(m),load(ART/'full_normal'/f'{e}_model.npz'));array_checks(model_arrays(m),load(f'artifacts/experiment{e}/normal_model.npz'));assert m.threshold==audit['variants'][e]['q99'];expected={k:np.concatenate([m.score(d)[k] for d in cal.values()]) for k in KEYS};array_checks(expected,load(ART/'full_normal'/f'{e}_scores.npz'));array_checks(expected,load(f'artifacts/experiment{e}/normal_calibration_scores.npz'))
    paired_checks(models,list(cal.values()))
    all_caches=[load_cache(ROOT,p.stem.split('_')[1],p.stem.split('_')[0]) for p in sorted(ROOT.glob('*.npz'))]
    for d in all_caches:causal_prefix(d)
    paired_checks(models,all_caches)
    score={e:{k:[] for k in KEYS} for e in VARIANTS};events={e:[] for e in VARIANTS};visual_events={e:[] for e in VARIANTS};fixed_events={e:[] for e in AFTER};labels=[];states=[];test_coverage=[];prediction_count=0
    for p in sorted(ROOT.glob('testing_*.npz')):
        seq=p.stem.split('_')[1];d=load_cache(ROOT,seq,'testing');results={e:m.score(d) for e,m in models.items()};n=int(d['frame_count']);y=evaluation_labels(np.load(Path('/media/jeong/ExtHDD/IPAD_vLLM/IPAD_dataset/R04/test_label')/f'{int(seq):03}.npy'),n);labels.append(y);states.append(hold_scores(d['indices'],evidence(d),n));test_coverage.append(coverage(d,seq,'test'))
        for e,r in results.items():
            saved=load(f'artifacts/experiment{e}/predictions/R04_{seq}.npz');np.testing.assert_array_equal(saved['labels'],y)
            for k in KEYS:np.testing.assert_array_equal(hold_scores(d['indices'],r[k],n),saved[k]);score[e][k].append(saved[k])
            np.testing.assert_array_equal(saved['object_scores'],np.concatenate([v for _,_,v in r['objects']]))
            for k in ['indices','phases','boxes','tracks']:np.testing.assert_array_equal(saved[k],d[k])
            events[e].extend([dict(v,sequence=seq) for v in anomaly_events(y,saved['combined']>models[e].threshold)]);visual_events[e].extend([dict(v,sequence=seq) for v in anomaly_events(y,saved['visual']>models[e].threshold)])
            if e in AFTER:fixed_events[e].extend([dict(v,sequence=seq) for v in anomaly_events(y,saved['combined']>models[PAIRS[e]].threshold)])
            prediction_count+=1
    y=np.concatenate(labels);state=np.concatenate(states);score={e:{k:np.concatenate(v) for k,v in r.items()} for e,r in score.items()};alarms={e:r['combined']>models[e].threshold for e,r in score.items()};summary={};branches={};pairs={};strata={}
    def counts(mask):return {'normal':int(np.sum(mask&(y==0))),'anomaly':int(np.sum(mask&(y==1)))}
    def event_diff(a,b):
        assert [(v['sequence'],v['start_frame']) for v in a]==[(v['sequence'],v['start_frame']) for v in b]
        return {'gained':[{'sequence':x['sequence'],'start_frame':x['start_frame']} for x,z in zip(a,b) if z['detected'] and not x['detected']],'lost':[{'sequence':x['sequence'],'start_frame':x['start_frame']} for x,z in zip(a,b) if x['detected'] and not z['detected']]}
    for e,s in score.items():
        q=models[e].threshold;mm={k:metrics(y,s[k]) for k in ['visual','process','combined']};official=json.loads(Path(f'results/experiment{e}/metrics.json').read_text());assert official['metrics']==mm;assert official['normal_q99_threshold']==q
        fpr=float(alarms[e][y==0].mean());recall=float(alarms[e][y==1].mean());assert fpr==official['test_normal_frame_alarm_rate'];assert recall==official['test_anomaly_frame_recall_at_q99']
        summary[e]={'metrics':mm,'q99':q,'alarms':counts(alarms[e]),'normal_fpr':fpr,'anomaly_recall':recall,'normal_holdout_fp':hold['totals'][e]['held_out_frame_alarms'],'events':summarize_events(events[e]),'transition_gate_open_frames':int(s['transition_valid'].sum()),'dwell_valid_frames':int(s['dwell_valid'].sum())}
        va=s['visual']>q;ta=s['transition_gated']>q;da=s['dwell_valid']&(s['dwell']>q);np.testing.assert_array_equal(alarms[e],va|ta|da)
        branches[e]={'visual_alarms':counts(va),'transition_only_alarms':counts(ta&~va&~da),'dwell_only_alarms':counts(da&~va&~ta),'both_process_only_alarms':counts(ta&da&~va),'process_added_over_visual':counts(alarms[e]&~va),'process_added_events':event_diff(visual_events[e],events[e]),'visual_events_at_combined_q99':summarize_events(visual_events[e])}
        strata[e]={name:{'frames':int(np.sum(state==i)),'label_counts':counts(state==i),'alarms':counts((state==i)&alarms[e])} for i,name in enumerate(STATE_NAMES)}
    for e in AFTER:
        b=PAIRS[e];fixed=score[e]['combined']>models[b].threshold;removed=alarms[b]&~alarms[e];added=alarms[e]&~alarms[b];changed=score[b]['combined']!=score[e]['combined'];removed_signal=score[b]['transition_gated']>score[e]['transition_gated']
        pairs[e]={'baseline':b,'added':counts(added),'removed':counts(removed),'changed_combined_frames':counts(changed),'closed_transition_signal_frames':counts(removed_signal),'closed_transition_alarms_at_baseline_q99':counts((state>=4)&(score[b]['transition_gated']>models[b].threshold)),'closed_boundary_still_alarm_at_own_q99':counts((state>=4)&alarms[e]),'event_changes':event_diff(events[b],events[e]),'same_new_scores_at_old_q99':{'q99':models[b].threshold,'alarms':counts(fixed),'events':summarize_events(fixed_events[e]),'own_q99_added':counts(alarms[e]&~fixed),'own_q99_removed':counts(fixed&~alarms[e])},'states':{name:{'removed':counts(removed&(state==i)),'added':counts(added&(state==i)),'changed_combined':counts(changed&(state==i))} for i,name in enumerate(STATE_NAMES)}}
        if models[e].threshold==models[b].threshold:assert not added.any();assert not np.any(removed&(state<4))
    write(OUT/'diagnostic.json',{'variants':summary,'paired_contrasts':pairs,'branch_contribution':branches,'evidence_strata':strata,'normal_holdout_changes':normal_changes,'normal_counterexamples':counterexamples,'test_coverage':test_coverage,'events':events,'note':'R04 repeated development. Fixed score components and references; only transition fusion gate changes and normal q99 is recomputed. Detection delay is conditional on detected GT intervals. Same track IDs do not prove semantic identity.'})
    write(OUT/'validation.json',{'protocol_hashes_verified':True,'source_feature_files_checked':len(all_caches),'normal_holdout_cells_reconstructed':fold_count,'test_prediction_files_reconstructed':prediction_count,'old_normal_holdout_scores_reproduced':True,'old_full_normal_models_scores_and_test_predictions_reproduced':True,'unchanged_pca_appearance_transition_references_dwell_verified':True,'normal_references_and_q99_independently_reconstructed':True,'same_pair_gate_subset_and_causal_prefix_verified':True,'scores_metrics_events_and_threshold_effects_recomputed':True});print(json.dumps({'variants':summary,'pairs':pairs,'branches':branches},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
