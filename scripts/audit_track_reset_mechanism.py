"""Post-result attribution only: support changes, normal failures and legacy checks."""
import json
from pathlib import Path
import numpy as np
from ipad_vad.context_dwell import complete_context_runs
from ipad_vad.data import hold_scores
from experiment30_track_reset import OUT,ART,SPLIT,GROUPS,GATES,KEYS,load,load_cache,root,write


def main():
    diagnostic=json.loads((OUT/'diagnostic.json').read_text());support={};context_branches={};normal_failures=[]
    for g in GROUPS:
        buckets={}
        for s in SPLIT['fit']:
            for key,duration in complete_context_runs(load_cache(root(g),s)):
                b=buckets.setdefault(str(key),{'durations':[],'videos':[]});b['durations'].append(duration);b['videos'].append(s)
        support[g]={k:{'complete_runs':len(v['durations']),'distinct_videos':len(set(v['videos'])),'videos':sorted(set(v['videos'])),'min_median_max_frames':[min(v['durations']),float(np.median(v['durations'])),max(v['durations'])],'supported':len(v['durations'])>=10} for k,v in sorted(buckets.items())}
    # Common rank turned out equal to experiment27. Check historical scores/models exactly.
    legacy_count=0
    for gate in GATES:
        e=f'30_control_{gate}';a=load(f'artifacts/experiment{e}/normal_model.npz');b=load(f'artifacts/experiment27_{gate}/normal_model.npz');assert a.keys()==b.keys()
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        for p in sorted(Path(f'artifacts/experiment{e}/predictions').glob('*.npz')):
            a=load(p);b=load(Path(f'artifacts/experiment27_{gate}/predictions')/p.name)
            for k in a:np.testing.assert_array_equal(a[k],b[k])
            legacy_count+=1
    for gate in GATES:
        e=f'30_reset_{gate}';q=diagnostic['variants'][e]['q99'];contexts={};dwell_fp_runs=[]
        for p in sorted(Path(f'artifacts/experiment{e}/predictions').glob('*.npz')):
            d=load(p);n=len(d['labels']);phase=hold_scores(d['indices'],d['phases'],n);y=d['labels'];va=d['visual']>q;ta=d['transition_gated']>q;da=d['dwell_valid']&(d['dwell']>q)
            for previous,state in [(1,3),(3,1)]:
                mask=d['dwell_valid']&(d['dwell_entry_context']==previous)&(phase==state);row=contexts.setdefault(f'{previous}->{state}',{'valid_normal':0,'valid_anomaly':0,'dwell_only_normal':0,'dwell_only_anomaly':0,'dwell_alarm_normal':0,'dwell_alarm_anomaly':0})
                for name,label in [('normal',0),('anomaly',1)]:
                    row['valid_'+name]+=int(np.sum(mask&(y==label)));row['dwell_only_'+name]+=int(np.sum(mask&da&~va&~ta&(y==label)));row['dwell_alarm_'+name]+=int(np.sum(mask&da&(y==label)))
            mask=da&~va&~ta&(y==0);edges=np.diff(np.r_[False,mask,False].astype(int))
            for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):dwell_fp_runs.append({'sequence':p.stem,'start':int(start),'end_exclusive':int(end),'entry_context':int(d['dwell_entry_context'][start]),'phase':int(phase[start]),'age_at_start':float(d['dwell_age'][start])})
        context_branches[e]={'contexts':contexts,'dwell_only_false_alarm_intervals':dwell_fp_runs}
    # New normal transition errors are examined at the holdout parameters used to generate them.
    for frame in [76,252]:
        d=load_cache(root('reset'),'02');i=int(np.flatnonzero(d['indices']==frame)[0]);refs=load(ART/'normal_holdout/30_reset_hold_exclude_02_model.npz');scores=load(ART/'normal_holdout/30_reset_hold_exclude_02_scores.npz');previous=int(d['phases'][i-1]);current=int(d['phases'][i]);reference=refs[f'process_reference_state_{previous}'];used=[s for s in SPLIT['calibration'] if s!='02'];counts=0;observed=0
        for seq in used:
            x=load_cache(root('reset'),seq);edge=(x['phases'][:-1]==previous)&(x['phases'][1:]==current);counts+=int(edge.sum());observed+=int(np.sum(edge&x['relation_valid'][:-1]&x['relation_valid'][1:]))
        normal_failures.append({'held_out':'02','frame':frame,'edge':[previous,current],'raw_transition':float(scores['transition_raw'][i]),'previous_state_reference_samples':len(reference),'reference_max':float(reference.max()),'remaining_calibration_same_edge_all':counts,'remaining_calibration_same_edge_observed':observed,'calibrated_transition':float(scores['transition'][i]),'selected_previous_pair':d['tracks'][d['relation_detection_indices'][i-1]].tolist(),'selected_current_pair':d['tracks'][d['relation_detection_indices'][i]].tolist()})
    write(OUT/'mechanism_audit.json',{'post_result_diagnostic':True,'no_parameter_or_score_changes':True,'historical27_models_exact':True,'historical27_prediction_files_exact':legacy_count,'fit_complete_context_support':support,'reset_context_branch_contribution':context_branches,'new_normal_transition_failures':normal_failures});print(json.dumps({'support':support,'context_branches':context_branches,'normal_failures':normal_failures},indent=2))


if __name__=='__main__':main()
