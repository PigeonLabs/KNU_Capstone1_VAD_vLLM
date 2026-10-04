"""Experiment27 normal-only preflight and reusable paired invariants."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from evaluate_route_holdout import load_cache,normal_model_arrays
from audit_missing_age import load,sha
from audit_bank_dispatch import verify_references

BEFORE=['26_hold','26_pool','26_age'];AFTER=['27_hold','27_pool','27_age'];VARIANTS=BEFORE+AFTER;PAIRS=dict(zip(AFTER,BEFORE))
PRESERVED=['visual','transition','dwell','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']
EXTRA=['transition_raw','transition_valid','transition_gated']
KEYS=['visual','process','combined',*PRESERVED[1:]]


def direct_mask(d):
    return np.array([i>0 and bool(d['relation_valid'][i-1]) and bool(d['relation_valid'][i]) for i in range(len(d['phases']))])


def evidence_state(d):
    # Mutually exclusive: first, observed pair, missing, reacquired.
    state=np.full(len(d['phases']),2,dtype=np.int8)
    if len(state):state[0]=0
    for i in range(1,len(state)):
        if d['relation_valid'][i]:state[i]=1 if d['relation_valid'][i-1] else 3
    return state


def fitted_models(fit):
    models={};source=load('artifacts/experiment26_hold/normal_model.npz')
    for e in VARIANTS:
        cfg=json.loads(Path(f'configs/experiment{e}.json').read_text());m=LognormalDwellBaseline(cfg,load_process(cfg));m.fit(fit);models[e]=m
        for (role,phase),s in m.spaces.items():
            for k in ['mean','basis']:np.testing.assert_array_equal(getattr(s,k),source[f'{k}_{role}_{phase}'])
        np.testing.assert_array_equal(m.transition,models[BEFORE[0]].transition)
        assert m.dwell.log_parameters==models[BEFORE[0]].dwell.log_parameters
    return models


def paired_checks(models,caches):
    for e in AFTER:
        new,old=models[e],models[PAIRS[e]];nr,oldr=normal_model_arrays(new),normal_model_arrays(old);assert nr.keys()==oldr.keys()
        for key in nr:
            if key!='threshold':np.testing.assert_array_equal(nr[key],oldr[key])
        for d in caches:
            a,b=old.score(d),new.score(d);mask=direct_mask(d)
            for key in PRESERVED:np.testing.assert_array_equal(a[key],b[key])
            for (_,_,x),(_,_,y) in zip(a['objects'],b['objects']):np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(old.raw(d)[1],b['transition_raw']);np.testing.assert_array_equal(mask,b['transition_valid']);np.testing.assert_array_equal(np.where(mask,a['transition'],0),b['transition_gated'])
            expected=np.maximum(b['transition_gated'],np.where(a['dwell_valid'],a['dwell'],0))
            np.testing.assert_array_equal(expected,b['process']);np.testing.assert_array_equal(np.maximum(a['visual'],expected),b['combined']);assert np.all(b['process']<=a['process']) and np.all(b['combined']<=a['combined'])


def main():
    out=Path('results/experiment27');art=Path('artifacts/experiment27/full_normal');art.mkdir(parents=True,exist_ok=True);root=Path('artifacts/experiment27/features/R04');split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];fit=[load_cache(root,s) for s in split['fit']];cal=[load_cache(root,s) for s in split['calibration']];rows={}
    with threadpool_limits(limits=4):
        models=fitted_models(fit)
        for e,m in models.items():
            m.calibrate(cal);verify_references(m,cal);r=[m.score(d) for d in cal];scores={k:np.concatenate([v[k] for v in r]) for k in KEYS+(EXTRA if e in AFTER else [])};scores['sequence']=np.concatenate([np.full(len(d['indices']),s) for s,d in zip(split['calibration'],cal)])
            np.savez_compressed(art/f'{e}_scores.npz',**scores);np.savez_compressed(art/f'{e}_references.npz',**normal_model_arrays(m))
            finite=all(np.isfinite(scores[k]).all() for k in ['visual','process','combined']);bounds=all(np.all((scores[k]>=0)&(scores[k]<=1)) for k in ['visual','process','combined'])
            rows[e]={'normal_q99':m.threshold,'finite':finite,'unit_interval':bounds,'alarm_not_structurally_blocked':bool(finite and bounds and m.threshold<1),'combined_at_one':int(np.sum(scores['combined']==1)),'sample_alarms':int(np.sum(scores['combined']>m.threshold)),'samples':len(scores['combined']),'previous_state_reference_counts':{str(k):len(v) for k,v in m.state_process_references.items()},'observed_pair_samples':sum(int(direct_mask(d).sum()) for d in cal)}
            if e in BEFORE:
                saved=load(f'artifacts/experiment{e}/normal_calibration_scores.npz')
                for k,v in scores.items():np.testing.assert_array_equal(v,saved[k])
        paired_checks(models,fit+cal)
    (out/'normal_audit.json').write_text(json.dumps({'normal_only':True,'variants':rows,'paired_pca_appearance_references_ungated_transition_dwell_preserved':True,'old_calibration_exactly_reproduced':True,'gated_process_and_combined_never_increase_at_fixed_q':True},indent=2)+'\n');print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
