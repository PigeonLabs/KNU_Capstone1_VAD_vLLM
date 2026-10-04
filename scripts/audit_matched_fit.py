"""Experiment22 normal-only counts, independent selection reconstruction and preflight."""
import hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.scoring import observations,Subspace
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from evaluate_route_holdout import load_cache,normal_model_arrays
from audit_factorial_normal import bank_usage

VARIANTS=['18','21_fit',*[f'22_s{s}' for s in range(5)]]
PROCESS=['transition','dwell','process','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def populations(fit):
    x={};valid={};videos={}
    for d in fit:
        for role,frames,features in observations(d):
            for p in [-1,*sorted(np.unique(d['phases'][frames]))]:
                mask=np.ones(len(frames),bool) if p==-1 else d['phases'][frames]==p
                key=(role,int(p));x.setdefault(key,[]).append(features[mask]);valid.setdefault(key,[]).append(d['relation_valid'][frames][mask]);videos.setdefault(key,[]).extend([d['sequence_id']]*int(mask.sum()))
    return {k:np.concatenate(v) for k,v in x.items()},{k:np.concatenate(v) for k,v in valid.items()},videos


def verify_banks(model,fit,variant):
    x,valid,videos=populations(fit);rng=np.random.default_rng(int(variant[-1])) if variant.startswith('22_s') else None;rows=[];expected_keys=set()
    for key in sorted(x):
        source=x[key];target=int(valid[key].sum());chosen=np.arange(len(source))
        if key[1]>=0:
            if rng is not None:
                chosen=np.sort(rng.choice(len(source),target,replace=False));np.testing.assert_array_equal(chosen,model.sampling_indices[key])
                assert model.sampling_support[key]=={'population':len(source),'target':target}
            elif variant=='21_fit':chosen=np.flatnonzero(valid[key])
        assert len(np.unique(chosen))==len(chosen) and np.all((chosen>=0)&(chosen<len(source)))
        expected=source[chosen];supported=len(expected)>=10
        if supported:
            expected_keys.add(key);s=Subspace(expected,.95,32);actual=model.spaces[key];assert actual.n==len(expected)
            for attr in ['mean','basis']:np.testing.assert_array_equal(getattr(s,attr),getattr(actual,attr))
        else:assert key not in model.spaces
        rows.append({'role':key[0],'phase':key[1],'population':len(source),'observed_target':target,'selected':len(chosen),'selected_observed':int(valid[key][chosen].sum()),'selected_videos':len(set(np.asarray(videos[key])[chosen])),'supported':supported,'rank':model.spaces[key].rank if supported else None,'selection_sha256':hashlib.sha256(chosen.astype('<i8').tobytes()).hexdigest()})
    assert expected_keys==set(model.spaces)
    return rows


def main():
    out=Path('results/experiment22');art=Path('artifacts/experiment22/full_normal');art.mkdir(parents=True,exist_ok=True);root=Path('artifacts/experiment22/features/R04');split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];fit=[load_cache(root,s) for s in split['fit']];cal=[load_cache(root,s) for s in split['calibration']];models={};results={};rows={}
    with threadpool_limits(limits=4):
        for e in VARIANTS:
            cfg=json.loads(Path(f'configs/experiment{e}.json').read_text());m=LognormalDwellBaseline(cfg,load_process(cfg));m.fit(fit);banks=verify_banks(m,fit,e);m.calibrate(cal);models[e]=m;results[e]=[m.score(d) for d in cal];assert m.route_calibration is None
            branches=['visual','transition','dwell','process','combined'];scores={k:np.concatenate([r[k] for r in results[e]]) for k in ['visual','combined',*PROCESS]};scores['sequence']=np.concatenate([np.full(len(d['indices']),s) for s,d in zip(split['calibration'],cal)])
            np.savez_compressed(art/f'{e}_scores.npz',**scores);np.savez_compressed(art/f'{e}_references.npz',**normal_model_arrays(m))
            np.savez_compressed(art/f'{e}_selections.npz',**{f'{r}_{p}':idx for (r,p),idx in m.sampling_indices.items()})
            finite=all(np.isfinite(scores[k]).all() for k in branches);bounds=all(np.all((scores[k]>=0)&(scores[k]<=1)) for k in branches);usage={}
            for d in cal:
                for k,v in bank_usage(m,d).items():usage[k]=usage.get(k,0)+v
            rows[e]={'normal_q99':m.threshold,'finite':finite,'unit_interval':bounds,'alarm_not_structurally_blocked':bool(finite and bounds and m.threshold<1),'combined_at_one':int(np.sum(scores['combined']==1)),'calibration_sample_alarms':int(np.sum(scores['combined']>m.threshold)),'banks':banks,'calibration_bank_usage_observations':usage,'role_calibration_counts':{str(k):len(v) for k,v in m.calibration.items()}}
        for e,m in models.items():
            for k in m.spaces:
                if k[1]==-1:
                    for attr in ['mean','basis']:np.testing.assert_array_equal(getattr(m.spaces[k],attr),getattr(models['18'].spaces[k],attr))
            if e.startswith('22_s'):
                assert set(m.spaces)==set(models['21_fit'].spaces)
                for k,s in m.spaces.items():assert s.n==models['21_fit'].spaces[k].n
                assert rows[e]['calibration_bank_usage_observations']==rows['21_fit']['calibration_bank_usage_observations']
            for r,base in zip(results[e],results['18']):
                for key in PROCESS:np.testing.assert_array_equal(r[key],base[key])
        for e in ['18','21_fit']:
            with np.load(f'artifacts/experiment{e}/normal_calibration_scores.npz') as saved,np.load(art/f'{e}_scores.npz') as fresh:
                for k in fresh:np.testing.assert_array_equal(saved[k],fresh[k])
            assert rows[e]['normal_q99']==json.loads(Path(f'results/experiment{e}/metrics.json').read_text())['normal_q99_threshold']
    (out/'normal_audit.json').write_text(json.dumps({'normal_only':True,'fit_sequences':split['fit'],'variants':rows,'independent_selection_and_pca_reconstruction':True,'random_count_support_and_routing_match_observed_fit':True,'all_pooled_and_process_preserved':True,'legacy_full_calibration_reproduced':True,'calibration_samples':482,'note':'Indices identify the sequence-ordered concatenated FIT observations in each role/phase bank. Relation mask determines target count only for random controls. Five fixed seeds; no selection.'},indent=2)+'\n');print(json.dumps({e:{k:v for k,v in r.items() if k not in ['banks','calibration_bank_usage_observations']} for e,r in rows.items()},indent=2))


if __name__=='__main__':main()
