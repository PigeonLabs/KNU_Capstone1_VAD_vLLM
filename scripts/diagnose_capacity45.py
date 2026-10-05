"""Measure captured normal variance from saved PCA bases, without refitting or labels."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.scoring import observations
from experiment45_normal import RUNS,SCENES,cfg,factory,load
from candidates45_common import OUT,ART,guard,sha,write


def main():
    guard();manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text());rows=[]
    for scene in SCENES:
        for run in RUNS:
            model=factory(cfg(run,scene));buckets={}
            for seq in manifest['subsplits'][scene]['downstream_fit']:
                data=load(run,scene,seq);phases=model.appearance_phases(data,stage='fit')
                for role,frames,x in observations(data):
                    buckets.setdefault((role,-1),[]).append(x)
                    for phase in np.unique(phases[frames]):
                        if phase>=0:buckets.setdefault((role,int(phase)),[]).append(x[phases[frames]==phase])
            path=ART/'normal_models'/f'{run}_{scene}_full_model.npz'
            with np.load(path,allow_pickle=False) as f:
                for key in f:
                    if not key.startswith('mean_'):continue
                    role,phase=map(int,key.removeprefix('mean_').split('_'));suffix=f'{role}_{phase}'
                    x=np.concatenate(buckets[(role,phase)]).astype(np.float64);mean=f[key];basis=f['basis_'+suffix];n,rank=f['support_rank_'+suffix]
                    assert n==len(x) and rank==len(basis);np.testing.assert_allclose(x.mean(0),mean,rtol=0,atol=1e-14)
                    centered=x-mean;total=float(np.sum(centered**2));captured=float(np.sum((centered@basis.T)**2))
                    ratio=captured/total if total>0 else None
                    if ratio is not None:assert -1e-10<=ratio<=1+1e-10
                    rows.append({'run':run,'scene':scene,'role':role,'phase':phase,'samples':int(n),
                        'feature_dimension':x.shape[1],'rank':int(rank),'at_rank_cap':int(rank)==cfg(run,scene)['pca_max_rank'],
                        'captured_normal_variance_fraction':ratio,'below_target_variance':ratio is not None and ratio<cfg(run,scene)['pca_variance']-1e-8})
            print('Capacity checked',scene,run,flush=True)
    summary=[]
    for run in RUNS:
        group=[r for r in rows if r['run']==run];ratios=[r['captured_normal_variance_fraction'] for r in group if r['captured_normal_variance_fraction'] is not None]
        summary.append({'run':run,'spaces':len(group),'at_rank_cap':sum(r['at_rank_cap'] for r in group),
            'below_95pct':sum(r['below_target_variance'] for r in group),
            'unweighted_mean_variance_fraction':float(np.mean(ratios)),'minimum_variance_fraction':min(ratios)})
    write(OUT/'capacity_diagnostics.json',{'normal_only':True,'refitted':False,'label_access':False,'rows':rows,'summary':summary,
        'source_normal_checkpoint_sha256':sha(OUT/'normal_models_checkpoint.json'),'script_sha256':sha(__file__),
        'interpretation':'Measured retained FIT variance, not anomaly quality or proof that increasing rank helps; weak phase/box errors remain possible.'})


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
