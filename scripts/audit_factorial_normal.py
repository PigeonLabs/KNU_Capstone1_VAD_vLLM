"""Normal-only full calibration and bank audit for experiment21's four cells."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from ipad_vad.scoring import observations
from evaluate_route_holdout import load_cache,normal_model_arrays

VARIANTS=['18','21_fit','21_infer','19']


def bank_usage(model,d):
    result={};phases=model.appearance_phases(d)
    for role,frames,_ in observations(d):
        for i in frames:
            phase=int(phases[i]);key=model.appearance_space_key(role,phase);observed=bool(d['relation_valid'][i])
            reason='explicit_missing_pool' if phase<0 else 'support_pool_observed' if key[1]<0 and observed else 'support_pool_unobserved' if key[1]<0 else 'phase_observed' if observed else 'phase_unobserved'
            name=f'{role}:{reason}';result[name]=result.get(name,0)+1
    return result


def main():
    out=Path('results/experiment21');art=Path('artifacts/experiment21/full_normal');art.mkdir(parents=True,exist_ok=True);root=Path('artifacts/experiment21/features/R04');split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];fit=[load_cache(root,s) for s in split['fit']];cal=[load_cache(root,s) for s in split['calibration']];models={};results={};rows={}
    with threadpool_limits(limits=4):
        for e in VARIANTS:
            cfg=json.loads(Path(f'configs/experiment{e}.json').read_text());m=LognormalDwellBaseline(cfg,load_process(cfg));m.fit(fit);m.calibrate(cal);models[e]=m;results[e]=[m.score(d) for d in cal]
            assert m.route_calibration is None
            branches=['visual','transition','dwell','process','combined'];scores={k:np.concatenate([r[k] for r in results[e]]) for k in branches+['dwell_valid','dwell_age','dwell_reason','dwell_entry_context']};scores['sequence']=np.concatenate([np.full(len(d['indices']),s) for s,d in zip(split['calibration'],cal)])
            np.savez_compressed(art/f'{e}_scores.npz',**scores);np.savez_compressed(art/f'{e}_references.npz',**normal_model_arrays(m))
            finite=all(np.isfinite(scores[k]).all() for k in branches);bounds=all(np.all((scores[k]>=0)&(scores[k]<=1)) for k in branches)
            usage={}
            for d in cal:
                for k,v in bank_usage(m,d).items():usage[k]=usage.get(k,0)+v
            rows[e]={'normal_q99':m.threshold,'finite':finite,'unit_interval':bounds,'alarm_not_structurally_blocked':bool(finite and bounds and m.threshold<1),'combined_at_one':int(np.sum(scores['combined']==1)),'calibration_sample_alarms':int(np.sum(scores['combined']>m.threshold)),'subspaces':[{'role':r,'phase':p,'samples':s.n,'rank':s.rank} for (r,p),s in sorted(m.spaces.items())],'calibration_bank_usage_observations':usage,'role_calibration_counts':{str(k):len(v) for k,v in m.calibration.items()}}
        for left,right in [('18','21_infer'),('19','21_fit')]:
            assert set(models[left].spaces)==set(models[right].spaces)
            for k in models[left].spaces:
                for attr in ['mean','basis']:np.testing.assert_array_equal(getattr(models[left].spaces[k],attr),getattr(models[right].spaces[k],attr))
        for e in VARIANTS:
            for k in models[e].spaces:
                if k[1]==-1:
                    for attr in ['mean','basis']:np.testing.assert_array_equal(getattr(models[e].spaces[k],attr),getattr(models['18'].spaces[k],attr))
            for r,base in zip(results[e],results['18']):
                for key in ['transition','dwell','process','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']:np.testing.assert_array_equal(r[key],base[key])
        for e in ['18','19']:
            with np.load(f'artifacts/experiment{e}/normal_calibration_scores.npz') as saved,np.load(art/f'{e}_scores.npz') as fresh:
                for k in fresh:np.testing.assert_array_equal(saved[k],fresh[k])
            assert rows[e]['normal_q99']==json.loads(Path(f'results/experiment{e}/metrics.json').read_text())['normal_q99_threshold']
    (out/'normal_audit.json').write_text(json.dumps({'normal_only':True,'variants':rows,'same_A_same_pca_verified':True,'all_pooled_and_process_preserved':True,'legacy_full_calibration_reproduced':True,'calibration_samples':482,'note':'Four prespecified cells. No route-specific CDF and no configuration selection.'},indent=2)+'\n');print(json.dumps({e:{k:v for k,v in r.items() if k not in ['subspaces','calibration_bank_usage_observations']} for e,r in rows.items()},indent=2))


if __name__=='__main__':main()
