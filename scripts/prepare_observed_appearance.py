"""Reuse experiment18 caches unchanged and audit normal-only appearance banks."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from ipad_vad.scoring import observations


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def usage(model,data):
    counts={};phases=model.appearance_phases(data)
    for role,frames,_ in observations(data):
        for i in frames:
            phase=int(phases[i]);key=model.appearance_space_key(role,phase)
            reason='unobserved_pool' if phase<0 else 'observed_phase' if key==(role,phase) else 'observed_support_pool'
            label=f'{role}:{reason}';counts[label]=counts.get(label,0)+1
    return counts


def main():
    out=Path('results/experiment19');out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path('configs/experiment19.json').read_text());split=json.loads(Path('results/stage00/splits.json').read_text())['R04']
    source=Path('artifacts/experiment18/features/R04');target=Path('artifacts/experiment19');target.mkdir(parents=True,exist_ok=True)
    link=target/'features'
    if not link.exists():link.symlink_to('../experiment18/features',target_is_directory=True)
    assert (link/'R04').resolve()==source.resolve()
    files=[Path('configs/experiment19.json'),Path('results/stage00/splits.json'),Path(cfg['process_discovery']),Path('docs/EXPERIMENT19_PLAN.md'),*sorted(Path('src/ipad_vad').glob('*.py')),Path(__file__).relative_to(Path.cwd()),Path('scripts/evaluate_baseline.py'),Path('scripts/check_normal_feasibility.py')]
    protocol=out/'pre_normal_protocol.json'
    if protocol.exists():
        saved=json.loads(protocol.read_text())
        for p,h in saved['file_sha256'].items():assert sha(p)==h,p
        for p,h in saved['source_features_sha256'].items():assert sha(source/p)==h,p
    else:
        protocol.write_text(json.dumps({'created_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'Before experiment19 normal fit/calibration and test scoring','file_sha256':{str(p):sha(p) for p in files},'source_features_sha256':{p.name:sha(p) for p in sorted(source.glob('*.npz'))},'decision':'Only appearance bank conditioning changes; no parameter search; all experiment18 feature arrays reused byte-for-byte.'},indent=2)+'\n')
    fit=[load(source/f'training_{s}.npz') for s in split['fit']]
    old_cfg=json.loads(Path('configs/experiment18.json').read_text())
    with threadpool_limits(limits=4):
        model=LognormalDwellBaseline(cfg,load_process(cfg));model.fit(fit)
        old=LognormalDwellBaseline(old_cfg,load_process(old_cfg));old.fit(fit)
    saved_old=load('artifacts/experiment18/normal_model.npz')
    for (role,phase),space in old.spaces.items():
        np.testing.assert_array_equal(space.mean,saved_old[f'mean_{role}_{phase}']);np.testing.assert_array_equal(space.basis,saved_old[f'basis_{role}_{phase}'])
    support={};routes={}
    for d in fit:
        for role,frames,x in observations(d):
            for p in range(model.k):
                mask=d['phases'][frames]==p;key=f'{role}:{p}';row=support.setdefault(key,{'role':role,'phase':p,'all':0,'observed':0})
                row['all']+=int(mask.sum());row['observed']+=int(np.sum(mask&d['relation_valid'][frames]))
        for k,v in usage(model,d).items():routes[k]=routes.get(k,0)+v
    for key,space in model.spaces.items():
        if key[1]==-1:
            np.testing.assert_array_equal(space.mean,old.spaces[key].mean);np.testing.assert_array_equal(space.basis,old.spaces[key].basis);assert space.n==old.spaces[key].n
    np.testing.assert_array_equal(model.transition,old.transition)
    for key in model.dwell.context_durations:np.testing.assert_array_equal(model.dwell.context_durations[key],old.dwell.context_durations[key])
    result={'normal_fit_only':True,'all_features_reused':True,'legacy_fit_reproduces_saved_experiment18':True,'pooled_subspaces_exactly_preserved':True,'transition_and_dwell_fit_preserved':True,'role_phase_support':list(support.values()),'fit_bank_usage_observations':routes,'old_phase_bank_count':sum(p>=0 for _,p in old.spaces),'new_phase_bank_count':sum(p>=0 for _,p in model.spaces),'new_subspaces':[{'role':r,'phase':p,'samples':s.n,'rank':s.rank} for (r,p),s in sorted(model.spaces.items())],'note':'Usage counts are feature observations (global plus crops), not distinct source frames. Unobserved samples remain in each role pooled bank once.'}
    (out/'normal_appearance_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
