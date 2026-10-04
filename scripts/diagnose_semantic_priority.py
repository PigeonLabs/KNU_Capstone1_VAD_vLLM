"""Reconstruct experiment34 scores, frozen normal holdouts and paired contributions."""
import copy,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.data import hold_scores,evaluation_labels
from ipad_vad.events import anomaly_events,summarize_events
from ipad_vad.scoring import empirical_percentile
from evaluate_baseline import metrics
from experiment34_semantic_priority import OUT,ART,SOURCE,GROUPS,GATES,VARIANTS,SPLIT,write,cfg,root,load,models as relation_models,configure_common,independently_check
from experiment30_track_reset import KEYS,verify_freeze,arrays,load_cache,LognormalDwellBaseline,load_process,verify_references
from ipad_vad.transition_evidence import same_track_pair_transition_mask


def assert_arrays(expected,saved):
    for k,v in expected.items():np.testing.assert_array_equal(v,saved[k],err_msg=k)


def process_audit(m,caches):
    refs=[];states={i:[] for i in range(4)}
    for d in caches:
        p=d['phases'];raw=np.zeros(len(p));raw[1:]=-np.log(m.transition[p[:-1],p[1:]])+(~m.allowed[p[:-1],p[1:]])
        refs.extend(raw)
        for state in states:states[state].extend(raw[1:][p[:-1]==state])
    np.testing.assert_array_equal(m.process_reference,refs)
    for state,r in states.items():np.testing.assert_array_equal(r,m.state_process_references[state])
    for d in caches:
        r=m.score(d);p=d['phases'];expected=[]
        for i,x in enumerate(r['transition_raw']):
            ref=states[p[i-1]] if i and len(states[p[i-1]])>=10 else refs
            ref=np.array(ref);expected.append((np.sum(ref<x)+.5*np.sum(ref==x))/len(ref))
        gate=same_track_pair_transition_mask(d)
        np.testing.assert_array_equal(r['transition'],expected);np.testing.assert_array_equal(r['transition_gated'],np.where(gate,expected,0))


def main():
    for name in ['pre_normal_protocol.json','pre_calibration_checkpoint.json','pre_test_checkpoint.json','normal_verification_checkpoint.json','pre_evaluation_review_checkpoint.json','test_input_checkpoint.json']:verify_freeze(name)
    audit=json.loads((OUT/'normal_audit.json').read_text());eligible=audit['eligible'];models={};normal_errors=[];trace=[];fold_count=0
    for e in eligible:
        g=e.split('_')[1];options=cfg(e);fit=[load_cache(root(g),s) for s in SPLIT['fit']];cal={s:load_cache(root(g),s) for s in SPLIT['calibration']};base=LognormalDwellBaseline(options,load_process(options));base.fit(fit)
        for row in [r for r in audit['folds'] if r['variant']==e]:
            held=row['held_out'];used=[cal[s] for s in cal if s!=held];assert row['calibration_sequences']==[s for s in cal if s!=held]
            m=copy.deepcopy(base);m.calibrate(used);verify_references(m,used);process_audit(m,used);assert_arrays(arrays(m),load(ART/'normal_holdout'/f'{e}_exclude_{held}_model.npz'));r=m.score(cal[held]);assert_arrays({k:r[k] for k in KEYS},load(ART/'normal_holdout'/f'{e}_exclude_{held}_scores.npz'));assert m.threshold==row['q99'];dense=hold_scores(cal[held]['indices'],r['combined']>m.threshold,int(cal[held]['frame_count']));assert int(dense.sum())==row['held_out_frame_alarms'];fold_count+=1
            for i in np.flatnonzero(r['combined']>m.threshold):normal_errors.append({'variant':e,'sequence':held,'source_frame':int(cal[held]['indices'][i]),'phase':int(cal[held]['phases'][i]),'previous_phase':int(cal[held]['phases'][i-1]) if i else None,'q99':m.threshold,**{k:float(r[k][i]) for k in ['visual','transition','transition_gated','dwell','process','combined']}})
        m=copy.deepcopy(base);m.calibrate(list(cal.values()));verify_references(m,list(cal.values()));process_audit(m,list(cal.values()));assert_arrays(arrays(m),load(ART/'full_normal'/f'{e}_model.npz'));assert_arrays(arrays(m),load(f'artifacts/experiment{e}/normal_model.npz'));full={k:np.concatenate([m.score(d)[k] for d in cal.values()]) for k in KEYS};assert_arrays(full,load(ART/'full_normal'/f'{e}_scores.npz'));assert_arrays(full,load(f'artifacts/experiment{e}/normal_calibration_scores.npz'));models[e]=m
    # Exact inputs and transforms, including zero mixed-pair windows, were independently audited in preparation.
    # Confirm all arrays still match the frozen transform, and pooled models are identical.
    relations=relation_models();files=0
    for group in GROUPS:
        for p in sorted(root(group).glob('*.npz')):
            original=load(SOURCE/p.name);derived=load(p);phase,valid,chosen,desc=relations[group].transform(original);independently_check(original,relations[group],group,phase,valid,chosen,desc);np.testing.assert_array_equal(valid,original['relation_valid']);np.testing.assert_array_equal(chosen[:,1],original['relation_detection_indices'][:,1])
            for key in original:
                if key not in ['phases','relation_descriptors','relation_detection_indices']:np.testing.assert_array_equal(original[key],derived[key])
            assert_arrays(dict(phases=phase,relation_valid=valid,relation_detection_indices=chosen,relation_descriptors=desc),derived);files+=1
    for gate in GATES:
        a,b=f'34_control_{gate}',f'34_semantic_{gate}'
        if a in models and b in models:
            for (role,phase),s in models[a].spaces.items():
                if phase==-1:
                    z=models[b].spaces[role,phase];np.testing.assert_array_equal(s.mean,z.mean);np.testing.assert_array_equal(s.basis,z.basis)
    scores={e:{k:[] for k in KEYS} for e in eligible};labels=[];events={e:[] for e in eligible};visual_events={e:[] for e in eligible};fixed_events={e:[] for e in eligible};per_sequence=[];pred_count=0
    for p in sorted(SOURCE.glob('testing_*.npz')):
        seq=p.stem.split('_')[1];computed={}
        for e,m in models.items():
            d=load_cache(root(e.split('_')[1]),seq,'testing');r=m.score(d);computed[e]=(d,r)
        n=int(next(iter(computed.values()))[0]['frame_count']);y=evaluation_labels(np.load(Path('/media/jeong/ExtHDD/IPAD_vLLM/IPAD_dataset/R04/test_label')/f'{int(seq):03}.npy'),n);labels.append(y)
        for e,(d,r) in computed.items():
            saved=load(f'artifacts/experiment{e}/predictions/R04_{seq}.npz');np.testing.assert_array_equal(saved['labels'],y)
            if e.startswith('34_control_') and json.loads((OUT/'fit_rank_control.json').read_text())['control_rank_map_matches_31']:
                previous=load(f'artifacts/experiment31_{e.split("_")[-1]}/predictions/R04_{seq}.npz')
                assert set(saved)==set(previous)
                for key,value in saved.items():np.testing.assert_array_equal(value,previous[key],err_msg=key)
            for k in KEYS:np.testing.assert_array_equal(hold_scores(d['indices'],r[k],n),saved[k]);scores[e][k].append(saved[k])
            np.testing.assert_array_equal(np.concatenate([v for _,_,v in r['objects']]),saved['object_scores']);pred_count+=1
            def ev(branch,q):return [dict(v,sequence=seq) for v in anomaly_events(y,saved[branch]>q)]
            events[e].extend(ev('combined',models[e].threshold));visual_events[e].extend(ev('visual',models[e].threshold));paired=e.replace('semantic','control');fixed_events[e].extend(ev('combined',models[paired].threshold if paired in models else models[e].threshold))
            per_sequence.append({'variant':e,'sequence':seq,**metrics(y,saved['combined']),'fp':int(np.sum((y==0)&(saved['combined']>models[e].threshold))),'tp':int(np.sum((y==1)&(saved['combined']>models[e].threshold)))})
    y=np.concatenate(labels);scores={e:{k:np.concatenate(v) for k,v in s.items()} for e,s in scores.items()};summary={};pairs={};branches={}
    def counts(mask):return {'normal':int(np.sum(mask&(y==0))),'anomaly':int(np.sum(mask&(y==1)))}
    def differences(a,b):
        return {'gained':[{'sequence':x['sequence'],'start_frame':x['start_frame']} for x,z in zip(a,b) if z['detected'] and not x['detected']],'lost':[{'sequence':x['sequence'],'start_frame':x['start_frame']} for x,z in zip(a,b) if x['detected'] and not z['detected']]}
    alarms={e:s['combined']>models[e].threshold for e,s in scores.items()}
    for e,s in scores.items():
        q=models[e].threshold;ma=json.loads(Path(f'results/experiment{e}/metrics.json').read_text());mm={k:metrics(y,s[k]) for k in ['visual','process','combined']};assert mm==ma['metrics'];assert ma['normal_q99_threshold']==q
        summary[e]={'metrics':mm,'q99':q,'alarms':counts(alarms[e]),'normal_fpr':float(alarms[e][y==0].mean()),'anomaly_recall':float(alarms[e][y==1].mean()),'normal_holdout_fp':sum(r['held_out_frame_alarms'] for r in audit['folds'] if r['variant']==e),'events':summarize_events(events[e]),'dwell_valid_frames':int(s['dwell_valid'].sum()),'observed_pair_frames':int(s['transition_valid'].sum())}
        va=s['visual']>q;ta=s['transition_gated']>q;da=s['dwell_valid']&(s['dwell']>q);np.testing.assert_array_equal(alarms[e],va|ta|da)
        branches[e]={'visual_alarms':counts(va),'transition_only_alarms':counts(ta&~va&~da),'dwell_only_alarms':counts(da&~va&~ta),'both_process_only_alarms':counts(ta&da&~va),'process_added_over_visual':counts(alarms[e]&~va),'process_added_events':differences(visual_events[e],events[e]),'visual_events_at_combined_q99':summarize_events(visual_events[e])}
    for gate in GATES:
        b,e=f'34_control_{gate}',f'34_semantic_{gate}'
        if b not in models or e not in models:continue
        fixed=scores[e]['combined']>models[b].threshold
        pairs[gate]={'added':counts(alarms[e]&~alarms[b]),'removed':counts(alarms[b]&~alarms[e]),'semantic_at_control_q99':{'q99':models[b].threshold,'alarms':counts(fixed),'events':summarize_events(fixed_events[e])},'own_vs_control_q99_on_same_semantic_scores':{'added':counts(alarms[e]&~fixed),'removed':counts(fixed&~alarms[e])},'events':differences(events[b],events[e]),'changed_combined_frames':int(np.sum(scores[e]['combined']!=scores[b]['combined']))}
    write(OUT/'diagnostic.json',{'variants':summary,'paired_contrasts':pairs,'branch_contribution':branches,'normal_holdout_alarms':normal_errors,'events':events,'per_sequence':per_sequence,'note':'Same normal-FIT rank policy and fixed phase centers. Changed phases refit PCA/transition/dwell/CDF. All test statistics are repeated R04 development, not independent confirmation. Delay is conditional on detected GT intervals; time is source frames.'})
    write(OUT/'validation.json',{'frozen_protocols_verified':True,'normal_holdout_cells_reconstructed':fold_count,'test_predictions_reconstructed':pred_count,'derived_feature_files_checked':files,'pooled_pca_unchanged':True,'normal_reference_q99_same_pair_process_gate_independently_reconstructed':True,'labels_metrics_events_recomputed':True})
    print(json.dumps({'summary':summary,'paired_contrasts':pairs,'branches':branches},indent=2),flush=True)


if __name__=='__main__':
    configure_common()
    with threadpool_limits(limits=4):main()
