"""Experiment28 normal-only observed-transition calibration and independent reference checks."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from evaluate_route_holdout import load_cache,normal_model_arrays
from audit_missing_age import load,sha
from audit_bank_dispatch import verify_references
from audit_transition_gate import direct_mask,evidence_state,EXTRA,KEYS

BEFORE=['27_hold','27_pool','27_age'];AFTER=['28_hold','28_pool','28_age'];VARIANTS=BEFORE+AFTER;PAIRS=dict(zip(AFTER,BEFORE))
PRESERVED=['visual','transition_raw','transition_valid','dwell','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']


def fitted_models(fit):
    models={};source=load('artifacts/experiment27_hold/normal_model.npz')
    for e in VARIANTS:
        cfg=json.loads(Path(f'configs/experiment{e}.json').read_text());m=LognormalDwellBaseline(cfg,load_process(cfg));m.fit(fit);models[e]=m
        for (role,phase),space in m.spaces.items():
            for k in ['mean','basis']:np.testing.assert_array_equal(getattr(space,k),source[f'{k}_{role}_{phase}'])
        np.testing.assert_array_equal(m.transition,models[BEFORE[0]].transition);assert m.dwell.log_parameters==models[BEFORE[0]].dwell.log_parameters
    return models


def verify_process(m,caches):
    observed=m.cfg['process_calibration'].get('population')=='consecutive_observed';expected=[];states={s:[] for s in range(m.k)};support={str(s):{'samples':0,'videos':[]} for s in range(m.k)};global_videos=[]
    for d in caches:
        # Directly reconstruct transition surprise and grammar penalty from fixed FIT parameters.
        p=d['phases'];raw=np.zeros(len(p));raw[1:]=-np.log(m.transition[p[:-1],p[1:]])+(~m.allowed[p[:-1],p[1:]])
        mask=direct_mask(d) if observed else np.ones(len(p),bool);expected.extend(raw[mask]);
        if mask.any():global_videos.append(d['sequence_id'])
        for state in states:
            use=(p[:-1]==state)&mask[1:];states[state].extend(raw[1:][use]);support[str(state)]['samples']+=int(use.sum())
            if use.any():support[str(state)]['videos'].append(d['sequence_id'])
    expected=np.array(expected);np.testing.assert_array_equal(expected,m.process_reference)
    for state,values in states.items():np.testing.assert_array_equal(np.array(values),m.state_process_references[state])
    for d in caches:
        r=m.score(d);raw=r['transition_raw'];calibrated=[];p=d['phases']
        for i,x in enumerate(raw):
            ref=states[p[i-1]] if i>0 and len(states[p[i-1]])>=m.minimum_support else expected;ref=np.asarray(ref);calibrated.append((np.sum(ref<x)+.5*np.sum(ref==x))/len(ref))
        np.testing.assert_array_equal(calibrated,r['transition']);np.testing.assert_array_equal(np.where(direct_mask(d),calibrated,0),r['transition_gated'])
    return {'global':{'samples':len(expected),'videos':global_videos},'previous_state':support}


def paired_checks(models,caches):
    for e in AFTER:
        new,old=models[e],models[PAIRS[e]];a,b=normal_model_arrays(new),normal_model_arrays(old);assert a.keys()==b.keys()
        for k in a:
            if k!='threshold' and not k.startswith('process_reference'):np.testing.assert_array_equal(a[k],b[k])
        for d in caches:
            x,y=old.score(d),new.score(d)
            for key in PRESERVED:np.testing.assert_array_equal(x[key],y[key])
            for (_,_,v),(_,_,w) in zip(x['objects'],y['objects']):np.testing.assert_array_equal(v,w)
            np.testing.assert_array_equal(np.maximum(y['transition_gated'],np.where(y['dwell_valid'],y['dwell'],0)),y['process']);np.testing.assert_array_equal(np.maximum(y['visual'],y['process']),y['combined'])
            closed=~direct_mask(d);np.testing.assert_array_equal(x['process'][closed],y['process'][closed]);np.testing.assert_array_equal(x['combined'][closed],y['combined'][closed])


def main():
    out=Path('results/experiment28');art=Path('artifacts/experiment28/full_normal');art.mkdir(parents=True,exist_ok=True);root=Path('artifacts/experiment28/features/R04');split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];fit=[load_cache(root,s) for s in split['fit']];cal=[load_cache(root,s) for s in split['calibration']];rows={}
    with threadpool_limits(limits=4):
        models=fitted_models(fit)
        for e,m in models.items():
            m.calibrate(cal);verify_references(m,cal);support=verify_process(m,cal);rs=[m.score(d) for d in cal];scores={k:np.concatenate([r[k] for r in rs]) for k in KEYS+EXTRA};scores['sequence']=np.concatenate([np.full(len(d['indices']),s) for s,d in zip(split['calibration'],cal)])
            np.savez_compressed(art/f'{e}_scores.npz',**scores);np.savez_compressed(art/f'{e}_references.npz',**normal_model_arrays(m));finite=all(np.isfinite(scores[k]).all() for k in ['visual','process','combined']);bounds=all(np.all((scores[k]>=0)&(scores[k]<=1)) for k in ['visual','process','combined'])
            rows[e]={'normal_q99':m.threshold,'finite':finite,'unit_interval':bounds,'alarm_not_structurally_blocked':bool(finite and bounds and m.threshold<1),'combined_at_one':int(np.sum(scores['combined']==1)),'sample_alarms':int(np.sum(scores['combined']>m.threshold)),'samples':len(scores['combined']),'process_support':support,'observed_pair_samples':sum(int(direct_mask(d).sum()) for d in cal)}
            if e in BEFORE:
                saved=load(f'artifacts/experiment{e}/normal_calibration_scores.npz')
                for k,v in scores.items():np.testing.assert_array_equal(v,saved[k])
        paired_checks(models,fit+cal)
    (out/'normal_audit.json').write_text(json.dumps({'normal_only':True,'variants':rows,'all_paired_visual_raw_transition_gate_and_dwell_preserved':True,'references_independently_reconstructed':True,'old_calibration_exactly_reproduced':True},indent=2)+'\n');print(json.dumps({e:{k:v for k,v in r.items() if k!='process_support'} for e,r in rows.items()},indent=2))


if __name__=='__main__':main()
