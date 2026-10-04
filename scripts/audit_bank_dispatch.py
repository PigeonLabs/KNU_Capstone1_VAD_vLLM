"""Experiment26 unchanged references, actual-bank dispatch and normal preflight."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from ipad_vad.scoring import empirical_percentile
from evaluate_route_holdout import load_cache,normal_model_arrays
from audit_missing_age import load,sha,PROCESS,STATE_NAMES,direct_routes
from audit_factorial_normal import bank_usage
from audit_request_calibration import independent_references

BEFORE=['25_hold','25_pool','25_age'];AFTER=['26_hold','26_pool','26_age'];VARIANTS=[*BEFORE,*AFTER];PAIRS=dict(zip(AFTER,BEFORE))


def verify_references(model,caches):
    expected=independent_references(model,caches);actual=model.request_calibration.references;assert expected.keys()==actual.keys()
    for key,v in expected.items():np.testing.assert_array_equal(v,actual[key])
    for d in caches:
        result=model.score(d);phases=model.appearance_phases(d)
        for (role,frames,raw),(_,_,score) in zip(model.raw(d)[0],result['objects']):
            codes=[]
            for p in phases[frames]:
                key=(role,int(p))
                if key not in model.spaces:key=(role,-1)
                if key not in model.spaces:key=(-1,-1)
                codes.append(int(key[1]>=0) if model.cfg.get('appearance_calibration_dispatch')=='actual_bank' else int(p>=0))
            codes=np.array(codes);v=np.empty(len(raw))
            for code in [0,1]:
                mask=codes==code;v[mask]=empirical_percentile(expected[role,code],raw[mask])
            np.testing.assert_array_equal(v,score)
    assert model.threshold==np.quantile(np.concatenate([model.score(d)['combined'] for d in caches]),.99,method='higher')


def same_bank_checks(models,caches):
    result={}
    for left,right in [('26_hold','26_pool'),('26_hold','26_age'),('26_pool','26_age')]:
        total=changed_request=0
        for d in caches:
            lm,rm=models[left],models[right];lp=lm.appearance_phases(d);rp=rm.appearance_phases(d);ls,rs=lm.score(d),rm.score(d);lr,rr=lm.raw(d)[0],rm.raw(d)[0]
            for (role,frames,a),(_,_,b),(_,_,ar),(_,_,br) in zip(ls['objects'],rs['objects'],lr,rr):
                mask=np.array([lm.appearance_space_key(role,lp[i])==rm.appearance_space_key(role,rp[i]) for i in frames]);np.testing.assert_array_equal(a[mask],b[mask]);np.testing.assert_array_equal(ar[mask],br[mask]);total+=int(mask.sum());changed_request+=int(np.sum(mask&((lp[frames]>=0)!=(rp[frames]>=0))))
        result[f'{left}_to_{right}']={'same_bank_feature_comparisons':total,'different_request_same_bank':changed_request,'changed_calibrated_scores_on_same_bank':0}
    return result


def main():
    out=Path('results/experiment26');art=Path('artifacts/experiment26/full_normal');art.mkdir(parents=True,exist_ok=True);root=Path('artifacts/experiment26/features/R04');split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];fit=[load_cache(root,s) for s in split['fit']];cal=[load_cache(root,s) for s in split['calibration']];models={};rows={};reference=None;source=load('artifacts/experiment25_hold/normal_model.npz');gap=json.loads(Path('results/experiment24/fit_gap_profile.json').read_text())
    with threadpool_limits(limits=4):
        for e in VARIANTS:
            cfg=json.loads(Path(f'configs/experiment{e}.json').read_text());m=LognormalDwellBaseline(cfg,load_process(cfg));m.fit(fit);models[e]=m;assert set(m.spaces)==set(models['25_hold'].spaces)
            for (r,p),s in m.spaces.items():
                assert s.n==models['25_hold'].spaces[r,p].n
                for attr in ['mean','basis']:np.testing.assert_array_equal(getattr(s,attr),source[f'{attr}_{r}_{p}'])
            if e.endswith('age'):assert m.missing_age.report()==gap
            m.calibrate(cal);verify_references(m,cal);results=[m.score(d) for d in cal];scores={k:np.concatenate([r[k] for r in results]) for k in ['visual','combined',*PROCESS]};scores['sequence']=np.concatenate([np.full(len(d['indices']),s) for s,d in zip(split['calibration'],cal)])
            np.savez_compressed(art/f'{e}_scores.npz',**scores);np.savez_compressed(art/f'{e}_references.npz',**normal_model_arrays(m));usage={}
            for d in cal:
                for k,v in bank_usage(m,d).items():usage[k]=usage.get(k,0)+v
            branches=['visual','combined','process','transition','dwell'];finite=all(np.isfinite(scores[k]).all() for k in branches);bounds=all(np.all((scores[k]>=0)&(scores[k]<=1)) for k in branches)
            rows[e]={'normal_q99':m.threshold,'finite':finite,'unit_interval':bounds,'alarm_not_structurally_blocked':bool(finite and bounds and m.threshold<1),'combined_at_one':int(np.sum(scores['combined']==1)),'calibration_sample_alarms':int(np.sum(scores['combined']>m.threshold)),'calibration_bank_usage_observations':usage,'request_support':[{'role':r,'reference':q,**v} for (r,q),v in sorted(m.request_calibration.support.items())]}
            for (r,q),v in m.request_calibration.references.items():np.testing.assert_array_equal(v,source[f'request_calibration_{r}_{q}'])
            if reference is None:reference=scores
            else:
                for k in PROCESS:np.testing.assert_array_equal(reference[k],scores[k])
            if e in BEFORE:
                saved=load(f'artifacts/experiment{e}/normal_calibration_scores.npz')
                for k,v in scores.items():np.testing.assert_array_equal(saved[k],v)
        for e in AFTER:
            for d in fit+cal:
                phases=models[e].appearance_phases(d)
                for (r,f,v),(rr,ff,b),(_,_,score),(_,_,previous) in zip(models[e].raw(d)[0],models[PAIRS[e]].raw(d)[0],models[e].score(d)['objects'],models[PAIRS[e]].score(d)['objects']):
                    assert r==rr;np.testing.assert_array_equal(f,ff);np.testing.assert_allclose(v,b,rtol=1e-12,atol=1e-12)
                    changed=np.array([phases[i]>=0 and models[e].appearance_space_key(r,phases[i])[1]<0 for i in f]);np.testing.assert_array_equal(score[~changed],previous[~changed])
        invariance=same_bank_checks(models,cal)
    (out/'normal_audit.json').write_text(json.dumps({'normal_only':True,'variants':rows,'all_banks_and_process_preserved':True,'all_legacy_calibrations_reproduced':True,'both_reference_arrays_equal_legacy':True,'normal_reference_independently_reconstructed':True,'same_bank_invariance':invariance,'tau_frames':56},indent=2)+'\n');print(json.dumps({e:{k:v for k,v in r.items() if k not in ['calibration_bank_usage_observations','request_support']} for e,r in rows.items()},indent=2))


if __name__=='__main__':main()
