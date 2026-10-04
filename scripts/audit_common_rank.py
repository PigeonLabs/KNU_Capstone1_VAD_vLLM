"""Experiment23 normal-only audit of fixed samples and prefix-rank constraints."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from evaluate_route_holdout import load_cache,normal_model_arrays
from audit_matched_fit import sha,verify_banks,PROCESS
from audit_factorial_normal import bank_usage

BEFORE=['21_fit',*[f'22_s{s}' for s in range(5)]]
AFTER=['23_obs',*[f'23_s{s}' for s in range(5)]]
VARIANTS=[*BEFORE,*AFTER]
PAIRS=dict(zip(AFTER,BEFORE))


def read_npz(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def verify_pair(old,new,fit,source_id):
    original_rows=verify_banks(old,fit,source_id);assert set(old.spaces)==set(new.spaces)
    source_model=read_npz(f'artifacts/experiment{source_id}/normal_model.npz')
    for key,s in old.spaces.items():
        for name in ['mean','basis']:np.testing.assert_array_equal(getattr(s,name),source_model[f'{name}_{key[0]}_{key[1]}'])
    for (r,p),idx in new.sampling_indices.items():
        np.testing.assert_array_equal(idx,old.sampling_indices[r,p]);np.testing.assert_array_equal(idx,source_model[f'fit_selection_{r}_{p}'])
    rows=[]
    for key,space in new.spaces.items():
        ref=old.spaces[key];np.testing.assert_array_equal(space.mean,ref.mean);assert space.n==ref.n
        rank=ref.rank if key[1]<0 else new.cfg['appearance_phase_ranks'][f'{key[0]}:{key[1]}']
        assert space.rank==rank;np.testing.assert_array_equal(space.basis,ref.basis[:rank])
        rows.append({'role':key[0],'phase':key[1],'samples':space.n,'original_rank':ref.rank,'common_rank':rank,'changed':rank!=ref.rank})
    np.testing.assert_array_equal(new.transition,old.transition)
    return rows,original_rows


def verify_raw_pair(old,new,caches):
    changed=0;checked=0
    for d in caches:
        assert bank_usage(old,d)==bank_usage(new,d)
        for (r,f,a),(rr,ff,b) in zip(old.raw(d)[0],new.raw(d)[0]):
            assert r==rr;np.testing.assert_array_equal(f,ff)
            assert np.all(b>=a-1e-12)
            for i,p in enumerate(d['phases'][f]):
                key=new.appearance_space_key(r,p)
                if new.spaces[key].rank==old.spaces[key].rank:assert abs(b[i]-a[i])<=1e-12
            changed+=int(np.sum(np.abs(a-b)>1e-12));checked+=len(a)
    return {'observations_checked':checked,'raw_residual_increased':changed}


def main():
    out=Path('results/experiment23');art=Path('artifacts/experiment23/full_normal');art.mkdir(parents=True,exist_ok=True);root=Path('artifacts/experiment23/features/R04');split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];fit=[load_cache(root,s) for s in split['fit']];cal=[load_cache(root,s) for s in split['calibration']];models={};rows={};pair_rows={};old_audit=json.loads(Path('results/experiment22/normal_audit.json').read_text())['variants']
    with threadpool_limits(limits=4):
        # Fit all models, and verify ranks/selection BEFORE any new calibration.
        for e in VARIANTS:
            cfg=json.loads(Path(f'configs/experiment{e}.json').read_text());m=LognormalDwellBaseline(cfg,load_process(cfg));m.fit(fit);models[e]=m
        actual_common={f'{r}:{p}':min(models[e].spaces[r,p].rank for e in BEFORE) for r,p in models[BEFORE[0]].spaces if p>=0}
        for after,before in PAIRS.items():
            old,new=models[before],models[after];assert new.cfg['appearance_phase_ranks']==actual_common
            banks,original=verify_pair(old,new,fit,before);assert original==old_audit[before]['banks'];pair_rows[after]={'before':before,'banks':banks,'normal_raw_checks':verify_raw_pair(old,new,fit+cal)}
        reference=None
        for e,m in models.items():
            m.calibrate(cal);results=[m.score(d) for d in cal];assert m.route_calibration is None;scores={k:np.concatenate([r[k] for r in results]) for k in ['visual','combined',*PROCESS]};scores['sequence']=np.concatenate([np.full(len(d['indices']),s) for s,d in zip(split['calibration'],cal)])
            np.savez_compressed(art/f'{e}_scores.npz',**scores);np.savez_compressed(art/f'{e}_references.npz',**normal_model_arrays(m));np.savez_compressed(art/f'{e}_selections.npz',**{f'{r}_{p}':idx for (r,p),idx in m.sampling_indices.items()})
            finite=all(np.isfinite(scores[k]).all() for k in ['visual','combined','process','transition','dwell']);bounds=all(np.all((scores[k]>=0)&(scores[k]<=1)) for k in ['visual','combined','process','transition','dwell']);usage={}
            for d in cal:
                for k,v in bank_usage(m,d).items():usage[k]=usage.get(k,0)+v
            rows[e]={'normal_q99':m.threshold,'finite':finite,'unit_interval':bounds,'alarm_not_structurally_blocked':bool(finite and bounds and m.threshold<1),'combined_at_one':int(np.sum(scores['combined']==1)),'calibration_sample_alarms':int(np.sum(scores['combined']>m.threshold)),'calibration_bank_usage_observations':usage,'subspaces':[{'role':r,'phase':p,'samples':s.n,'rank':s.rank} for (r,p),s in sorted(m.spaces.items())]}
            if reference is None:reference=scores
            else:
                for k in PROCESS:np.testing.assert_array_equal(reference[k],scores[k])
            if e in BEFORE:
                saved=read_npz(f'artifacts/experiment{e}/normal_calibration_scores.npz')
                for k,v in scores.items():np.testing.assert_array_equal(v,saved[k])
                assert m.threshold==old_audit[e]['normal_q99']
    (out/'normal_audit.json').write_text(json.dumps({'normal_only':True,'common_rank_rule':'Minimum original normal FIT rank per supported phase bank across all six fixed selections','common_ranks':actual_common,'pairs':pair_rows,'variants':rows,'legacy_calibration_exact':True,'all_process_exact':True,'samples_support_routes_pooled_preserved':True},indent=2)+'\n');print(json.dumps({e:{k:v for k,v in r.items() if k not in ['subspaces','calibration_bank_usage_observations']} for e,r in rows.items()},indent=2))


if __name__=='__main__':main()
