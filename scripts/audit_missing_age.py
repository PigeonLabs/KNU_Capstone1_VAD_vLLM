"""Normal-only appearance route audit for fixed, immediate and age-limited fallback."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from ipad_vad.missing_age import MissingAge
from ipad_vad.scoring import observations
from ipad_vad.data import hold_scores
from evaluate_route_holdout import load_cache,normal_model_arrays
from audit_matched_fit import PROCESS,sha
from audit_factorial_normal import bank_usage

VARIANTS=['23_obs','24_pool','24_age']
STATE_NAMES=['observed','initial_missing','missing_within_tau','missing_beyond_tau']


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def direct_routes(data,variant,tau):
    # Independent prefix calculation; never inspect the next valid sample.
    phases=[];states=[];ages=[];last=None;phase=None
    for index,p,valid in zip(data['indices'],data['phases'],data['relation_valid']):
        if valid:last=int(index);phase=int(p)
        age=-1 if last is None else int(index)-last
        state=0 if valid else 1 if last is None else 2 if age<=tau else 3
        route=int(p) if variant=='23_obs' else int(p) if valid else -1
        if variant=='24_age':route=-1 if state in [1,3] else phase
        phases.append(route);states.append(state);ages.append(age)
    return np.array(phases),np.array(states),np.array(ages)


def verify_calibration(model,caches,e,tau):
    refs={}
    for d in caches:
        phases,_,_=direct_routes(d,e,tau);np.testing.assert_array_equal(phases,model.appearance_phases(d))
        for role,frames,x in observations(d):
            expected=np.empty(len(x))
            for p in np.unique(phases[frames]):
                key=(role,int(p))
                if key not in model.spaces:key=(role,-1)
                if key not in model.spaces:key=(-1,-1)
                mask=phases[frames]==p;expected[mask]=model.spaces[key].residual(x[mask])
            refs.setdefault(role,[]).append(expected)
    for role,parts in refs.items():np.testing.assert_array_equal(np.concatenate(parts),model.calibration[role])
    assert model.route_calibration is None
    assert model.threshold==np.quantile(np.concatenate([model.score(d)['combined'] for d in caches]),.99,method='higher')


def main():
    out=Path('results/experiment24');art=Path('artifacts/experiment24/full_normal');art.mkdir(parents=True,exist_ok=True);root=Path('artifacts/experiment24/features/R04');split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];fit=[load_cache(root,s) for s in split['fit']];cal=[load_cache(root,s) for s in split['calibration']];profile=json.loads((out/'fit_gap_profile.json').read_text());tau=profile['tau_frames'];models={};rows={};source=load('artifacts/experiment23_obs/normal_model.npz');reference=None
    with threadpool_limits(limits=4):
        for e in VARIANTS:
            cfg=json.loads(Path(f'configs/experiment{e}.json').read_text());m=LognormalDwellBaseline(cfg,load_process(cfg));m.fit(fit);models[e]=m
            assert set(m.spaces)==set(models['23_obs'].spaces)
            for (r,p),s in m.spaces.items():
                assert s.n==models['23_obs'].spaces[r,p].n
                for attr in ['mean','basis']:np.testing.assert_array_equal(getattr(s,attr),source[f'{attr}_{r}_{p}'])
            if e=='24_age':assert m.missing_age.report()==profile
            m.calibrate(cal);verify_calibration(m,cal,e,tau);results=[m.score(d) for d in cal];scores={k:np.concatenate([r[k] for r in results]) for k in ['visual','combined',*PROCESS]};scores['sequence']=np.concatenate([np.full(len(d['indices']),s) for s,d in zip(split['calibration'],cal)])
            np.savez_compressed(art/f'{e}_scores.npz',**scores);np.savez_compressed(art/f'{e}_references.npz',**normal_model_arrays(m));usage={};strata={name:{'samples':0,'frames':0,'alarms':0,'global_phase_samples':0,'global_pooled_samples':0} for name in STATE_NAMES}
            for data,res in zip(cal,results):
                _,states,_=direct_routes(data,e,tau);route=m.appearance_routes(data,-1,np.arange(len(states)));dense=hold_scores(data['indices'],states,int(data['frame_count']));alarm=hold_scores(data['indices'],res['combined']>m.threshold,len(dense)).astype(bool)
                for i,name in enumerate(STATE_NAMES):
                    mask=states==i;row=strata[name];row['samples']+=int(mask.sum());row['frames']+=int(np.sum(dense==i));row['alarms']+=int(np.sum((dense==i)&alarm));row['global_phase_samples']+=int(np.sum(mask&(route==1)));row['global_pooled_samples']+=int(np.sum(mask&(route==0)))
                for k,v in bank_usage(m,data).items():usage[k]=usage.get(k,0)+v
            branches=['visual','combined','process','transition','dwell'];finite=all(np.isfinite(scores[k]).all() for k in branches);bounds=all(np.all((scores[k]>=0)&(scores[k]<=1)) for k in branches)
            rows[e]={'normal_q99':m.threshold,'finite':finite,'unit_interval':bounds,'alarm_not_structurally_blocked':bool(finite and bounds and m.threshold<1),'combined_at_one':int(np.sum(scores['combined']==1)),'calibration_sample_alarms':int(np.sum(scores['combined']>m.threshold)),'calibration_bank_usage_observations':usage,'calibration_age_strata':strata}
            if reference is None:
                reference=scores;saved=load('artifacts/experiment23_obs/normal_calibration_scores.npz')
                for k,v in scores.items():np.testing.assert_array_equal(v,saved[k])
            else:
                for k in PROCESS:np.testing.assert_array_equal(reference[k],scores[k])
    (out/'normal_audit.json').write_text(json.dumps({'normal_only':True,'tau_frames':tau,'variants':rows,'all_banks_and_process_preserved':True,'baseline_calibration_reproduced':True,'fit_gap_profile_matches_frozen':True},indent=2)+'\n');print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
