"""Experiment25 normal-only full-population references and route-invariant scores."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from ipad_vad.scoring import observations,empirical_percentile
from evaluate_route_holdout import load_cache,normal_model_arrays
from audit_missing_age import load,sha,PROCESS,direct_routes,STATE_NAMES
from audit_factorial_normal import bank_usage

BEFORE=['23_obs','24_pool','24_age'];AFTER=['25_hold','25_pool','25_age'];VARIANTS=[*BEFORE,*AFTER];PAIRS=dict(zip(AFTER,BEFORE))


def independent_references(model,caches):
    parts={}
    for d in caches:
        for role,frames,x in observations(d):
            phase_values=np.empty(len(x));pool_values=None
            for p in [-1,*sorted(np.unique(d['phases'][frames]))]:
                key=(role,int(p))
                if key not in model.spaces:key=(role,-1)
                if key not in model.spaces:key=(-1,-1)
                mask=np.ones(len(x),bool) if p==-1 else d['phases'][frames]==p
                s=model.spaces[key];centered=np.asarray(x[mask],np.float64)-s.mean
                residual=np.maximum(np.sum(centered**2,axis=-1)-np.sum((centered@s.basis.T)**2,axis=-1),0)
                if p==-1:pool_values=residual
                else:phase_values[mask]=residual
            for req,val in [(0,pool_values),(1,phase_values)]:parts.setdefault((role,req),[]).append(val)
    return {k:np.concatenate(v) for k,v in parts.items()}


def verify_references(model,caches):
    expected=independent_references(model,caches);actual=model.request_calibration.references;assert set(expected)==set(actual)
    for key,v in expected.items():np.testing.assert_array_equal(v,actual[key])
    for d in caches:
        r=model.score(d)
        for (role,frames,residual),(rr,ff,score) in zip(model.raw(d)[0],r['objects']):
            assert rr==role;np.testing.assert_array_equal(frames,ff);requests=model.appearance_phases(d)[frames]>=0;v=np.empty(len(score))
            for req in [0,1]:
                mask=requests==req;v[mask]=empirical_percentile(expected[role,req],residual[mask])
            np.testing.assert_array_equal(v,score)
    assert model.threshold==np.quantile(np.concatenate([model.score(d)['combined'] for d in caches]),.99,method='higher')


def check_gate_invariance(models,caches):
    count=0
    for d in caches:
        results={e:models[e].score(d) for e in AFTER};requests={e:models[e].appearance_phases(d)>=0 for e in AFTER}
        for left,right in [('25_hold','25_pool'),('25_hold','25_age'),('25_pool','25_age')]:
            equal=requests[left]==requests[right]
            for (r,f,a),(rr,ff,b) in zip(results[left]['objects'],results[right]['objects']):
                assert r==rr;np.testing.assert_array_equal(f,ff);np.testing.assert_array_equal(a[equal[f]],b[equal[f]]);count+=int(equal[f].sum())
            for key in ['visual','combined']:np.testing.assert_array_equal(results[left][key][equal],results[right][key][equal])
    return count


def main():
    out=Path('results/experiment25');art=Path('artifacts/experiment25/full_normal');art.mkdir(parents=True,exist_ok=True);root=Path('artifacts/experiment25/features/R04');split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];fit=[load_cache(root,s) for s in split['fit']];cal=[load_cache(root,s) for s in split['calibration']];models={};rows={};reference=None;source=load('artifacts/experiment23_obs/normal_model.npz');gap=json.loads(Path('results/experiment24/fit_gap_profile.json').read_text())
    with threadpool_limits(limits=4):
        for e in VARIANTS:
            cfg=json.loads(Path(f'configs/experiment{e}.json').read_text());m=LognormalDwellBaseline(cfg,load_process(cfg));m.fit(fit);models[e]=m
            assert set(m.spaces)==set(models['23_obs'].spaces)
            for (r,p),s in m.spaces.items():
                assert s.n==models['23_obs'].spaces[r,p].n
                for attr in ['mean','basis']:np.testing.assert_array_equal(getattr(s,attr),source[f'{attr}_{r}_{p}'])
            if e.endswith('age'):assert m.missing_age.report()==gap
            m.calibrate(cal)
            if e in AFTER:verify_references(m,cal)
            results=[m.score(d) for d in cal];scores={k:np.concatenate([r[k] for r in results]) for k in ['visual','combined',*PROCESS]};scores['sequence']=np.concatenate([np.full(len(d['indices']),s) for s,d in zip(split['calibration'],cal)])
            np.savez_compressed(art/f'{e}_scores.npz',**scores);np.savez_compressed(art/f'{e}_references.npz',**normal_model_arrays(m));usage={}
            for d in cal:
                for k,v in bank_usage(m,d).items():usage[k]=usage.get(k,0)+v
            branches=['visual','combined','process','transition','dwell'];finite=all(np.isfinite(scores[k]).all() for k in branches);bounds=all(np.all((scores[k]>=0)&(scores[k]<=1)) for k in branches)
            rows[e]={'normal_q99':m.threshold,'finite':finite,'unit_interval':bounds,'alarm_not_structurally_blocked':bool(finite and bounds and m.threshold<1),'combined_at_one':int(np.sum(scores['combined']==1)),'calibration_sample_alarms':int(np.sum(scores['combined']>m.threshold)),'calibration_bank_usage_observations':usage,'request_support':[] if m.request_calibration is None else [{'role':r,'request':q,**v} for (r,q),v in sorted(m.request_calibration.support.items())]}
            if reference is None:reference=scores
            else:
                for k in PROCESS:np.testing.assert_array_equal(reference[k],scores[k])
            if e in BEFORE:
                saved=load(f'artifacts/experiment{e}/normal_calibration_scores.npz')
                for k,v in scores.items():np.testing.assert_array_equal(saved[k],v)
        for e in AFTER:
            for key,v in models[e].request_calibration.references.items():np.testing.assert_array_equal(v,models['25_hold'].request_calibration.references[key])
            for d in fit+cal:
                for (r,f,v),(rr,ff,b) in zip(models[e].raw(d)[0],models[PAIRS[e]].raw(d)[0]):
                    assert r==rr;np.testing.assert_array_equal(f,ff);np.testing.assert_allclose(v,b,rtol=1e-12,atol=1e-12)
        invariant=check_gate_invariance(models,cal)
    (out/'normal_audit.json').write_text(json.dumps({'normal_only':True,'variants':rows,'all_banks_and_process_preserved':True,'all_legacy_calibrations_reproduced':True,'identical_full_population_request_references':True,'same_request_calibrated_feature_comparisons':invariant,'normal_reference_independently_reconstructed':True,'tau_frames':gap['tau_frames'],'note':'Canonical per-policy batches eliminate gate-dependent floating-point tie differences. Both inference paths are currently computed; runtime overhead not benchmarked.'},indent=2)+'\n');print(json.dumps({e:{k:v for k,v in r.items() if k not in ['calibration_bank_usage_observations','request_support']} for e,r in rows.items()},indent=2))


if __name__=='__main__':main()
