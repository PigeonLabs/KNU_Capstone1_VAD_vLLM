"""Post-result NORMAL-only provenance audit; no new model, score or gate is selected."""
import json
from pathlib import Path
import numpy as np
from ipad_vad.context_dwell import complete_context_runs,observed_entry_context
from ipad_vad.dwell import observed_ages
from experiment31_pair_gate import OUT,ROOT,SPLIT,evidence,load_cache,write,load


def main():
    records=[];usage=[];supported={(1,3),(3,1)}
    for partition in ['fit','calibration']:
        for seq in SPLIT[partition]:
            d=load_cache(ROOT,seq);p=d['phases'];v=d['relation_valid'];state=evidence(d);starts=np.r_[0,np.flatnonzero(p[1:]!=p[:-1])+1];ends=np.r_[starts[1:],len(p)];eligible=[]
            for start,end in zip(starts,ends):
                if start==0 or end==len(p) or not v[start-1:end+1].all():continue
                key=(int(p[start-1]),int(p[start]));duration=float(d['indices'][end]-d['indices'][start]);eligible.append((key,duration));incoming=state[start]==3;within=bool(np.all(state[start+1:end]==3));outgoing=state[end]==3
                pairs=d['tracks'][d['relation_detection_indices'][start-1:end+1]]
                assert bool(np.all(pairs==pairs[0]))==bool(incoming and within and outgoing)
                records.append({'partition':partition,'sequence':seq,'context':f'{key[0]}->{key[1]}','start':int(d['indices'][start]),'end':int(d['indices'][end]),'duration':duration,'incoming_same_pair':bool(incoming),'within_run_same_pair':within,'outgoing_same_pair':bool(outgoing),'identity_continuous_complete':bool(incoming and within and outgoing)})
            assert eligible==list(complete_context_runs(d))
            context=observed_entry_context(d);age,known=observed_ages(d);supported_valid=known&np.array([(int(a),int(b)) in supported for a,b in zip(context,p)]);identity_ok=np.zeros(len(p),bool);entry_ok=False
            for i in range(1,len(p)):
                if not v[i] or not v[i-1]:entry_ok=False
                elif p[i]!=p[i-1]:entry_ok=state[i]==3
                elif state[i]!=3:entry_ok=False
                identity_ok[i]=entry_ok
            usage.append({'partition':partition,'sequence':seq,'dwell_valid_samples':int(supported_valid.sum()),'same_pair_since_observed_entry_samples':int(np.sum(supported_valid&identity_ok)),'identity_discontinuous_since_entry_samples':int(np.sum(supported_valid&~identity_ok))})
            if partition=='calibration':
                saved=load('artifacts/experiment31_hold/normal_calibration_scores.npz');mask=saved['sequence']==seq;np.testing.assert_array_equal(supported_valid,saved['dwell_valid'][mask])
    totals={}
    for part in ['fit','calibration']:
        totals[part]={}
        for context in sorted({x['context'] for x in records if x['partition']==part}):
            r=[x for x in records if x['partition']==part and x['context']==context];pure=[x for x in r if x['identity_continuous_complete']]
            totals[part][context]={'complete_runs':len(r),'videos':len({x['sequence'] for x in r}),'incoming_same_pair_runs':sum(x['incoming_same_pair'] for x in r),'within_run_same_pair_runs':sum(x['within_run_same_pair'] for x in r),'outgoing_same_pair_runs':sum(x['outgoing_same_pair'] for x in r),'identity_continuous_complete_runs':len(pure),'identity_continuous_videos':len({x['sequence'] for x in pure})}
    result={'normal_only':True,'post_result_provenance_diagnostic':True,'no_score_parameter_or_support_change_applied':True,'rule_under_inspection':'Same selected pair through incoming edge, within phase run, and outgoing edge for a complete training duration. For current inference provenance, same pair since observed entry; no future exit is inspected.','complete_run_support':totals,'current_dwell_provenance':usage,'runs':records,'caveat':'Track IDs are not identity ground truth. Counts do not decide a new support threshold and are not new anomaly scores.'};write(OUT/'normal_dwell_identity_audit.json',result);print(json.dumps({'support':totals,'usage':{p:{k:sum(x[k] for x in usage if x['partition']==p) for k in ['dwell_valid_samples','same_pair_since_observed_entry_samples','identity_discontinuous_since_entry_samples']} for p in ['fit','calibration']}},indent=2))


if __name__=='__main__':main()
