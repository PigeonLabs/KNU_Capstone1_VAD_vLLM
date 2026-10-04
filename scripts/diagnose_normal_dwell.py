"""Inspect normal-only dwell fit and calibration before test evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
from ipad_vad.dwell import NormalDwell
from ipad_vad.context_dwell import ContextDwell


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/experiment11.json'));args=p.parse_args()
    cfg=json.loads(args.config.read_text());scene=cfg['scene'];n=cfg['experiment'];split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    def load(seq):
        with np.load(f'artifacts/experiment{n}/features/{scene}/training_{seq}.npz') as d:return dict(d)
    fit=[load(s) for s in split['fit']];cal=[load(s) for s in split['calibration']]
    m=ContextDwell(**cfg['normal_dwell'],**cfg['dwell_context']) if 'dwell_context' in cfg else NormalDwell(**cfg['normal_dwell'])
    m.fit(fit);m.calibrate(cal);groups={}
    for name,caches in [('fit',fit),('calibration',cal)]:
        phases=np.concatenate([d['phases'] for d in caches]);raws=[m.raw(d) for d in caches];raw=np.concatenate([r[0] for r in raws]);valid=np.concatenate([r[1] for r in raws]);reason=np.concatenate([r[3] for r in raws])
        scores=np.concatenate([m.score(d)[0] for d in caches]);by_state=[]
        for state in range(cfg['relational_phase']['k']):
            use=valid&(phases==state);by_state.append({'state':state,'sampled_total':int((phases==state).sum()),'valid':int(use.sum()),
                'raw_min_median_max':np.quantile(raw[use],[0,.5,1]).tolist() if use.any() else None,
                'percentile_min_median_max':np.quantile(scores[use],[0,.5,1]).tolist() if use.any() else None})
        groups[name]={'samples':len(valid),'valid':int(valid.sum()),'valid_fraction':float(valid.mean()),'reason_counts':{str(i):int((reason==i).sum()) for i in range(4)},'by_state':by_state}
    out={'scene':scene,'source':'NORMAL FIT and held-out NORMAL calibration only; no test access','complete_run_support':m.support,
        'supported_durations':{str(k):v.tolist() for k,v in m.durations.items()},'calibration_reference_samples':len(m.reference),'groups':groups,
        'raw_score':'-log((1 + count(D >= observed_age)) / (1 + n))','calibration':'Global empirical midrank of valid normal dwell raw scores only.',
        'reason_codes':{'0':'valid','1':'relation_missing','2':'entry_not_observed','3':'unsupported_state'},
        'limitations':['Complete-run exclusion is not a censoring-aware survival estimator.','Cluster duration is not verified action duration.','Calibration samples are temporally correlated.']}
    if 'dwell_context' in cfg:
        out.update(context_distribution=m.distribution,complete_context_support=m.context_support,
                   supported_context_durations={f'{a}->{b}':v.tolist() for (a,b),v in m.context_durations.items()})
        out['reason_codes']['3']='unsupported_entry_context'
    Path(f'results/experiment{n}/normal_dwell_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
