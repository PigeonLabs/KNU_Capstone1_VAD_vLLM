"""Normal-only check for a configured lognormal dwell pipeline."""
import argparse,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);args=p.parse_args()
    cfg=json.loads(Path(f'configs/experiment{args.experiment}.json').read_text());assert cfg['dwell_score']=='fit_lognormal_cdf'
    split=json.loads(Path('results/stage00/splits.json').read_text())[cfg['scene']];root=Path(f'artifacts/experiment{args.experiment}')
    def load(seq):
        with np.load(root/'features'/cfg['scene']/f'training_{seq}.npz',allow_pickle=False) as f:data=dict(f)
        data['sequence_id']=f'{cfg["scene"]}/training_{seq}'
        return data
    fit=[load(s) for s in split['fit']];cal=[load(s) for s in split['calibration']]
    with threadpool_limits(limits=4):
        model=LognormalDwellBaseline(cfg,load_process(cfg));model.fit(fit);model.calibrate(cal);results=[model.score(d) for d in cal]
    scores={k:np.concatenate([r[k] for r in results]) for k in ['visual','transition','dwell','process','combined','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']}
    scores['sequence']=np.concatenate([np.full(len(d['indices']),s) for s,d in zip(split['calibration'],cal)])
    finite=all(np.isfinite(scores[k]).all() for k in ['visual','transition','dwell','process','combined']);bounds=all(np.all((scores[k]>=0)&(scores[k]<=1)) for k in ['visual','transition','dwell','process','combined'])
    assert model.threshold==np.quantile(scores['combined'],cfg['calibration_quantile'],method='higher')
    result={'normal_only':True,'calibration_samples':len(scores['combined']),'all_branch_scores_finite':finite,'all_branch_scores_in_unit_interval':bounds,'normal_q99':model.threshold,'alarm_not_structurally_blocked':bool(finite and bounds and model.threshold<1),'dwell_at_one':int(np.sum(scores['dwell']==1)),'combined_at_one':int(np.sum(scores['combined']==1)),'calibration_sample_alarms':int(np.sum(scores['combined']>model.threshold)),'supported_contexts':[f'{a}->{b}' for a,b in model.dwell.context_durations],'lognormal_parameters':{f'{a}->{b}':{'mu':mu,'sigma':sigma} for (a,b),(mu,sigma) in model.dwell.log_parameters.items()},'note':'Necessary feasibility check on normal FIT/calibration only. No parameter selection or guarantee of useful test detection.'}
    np.savez_compressed(root/'preflight_calibration_scores.npz',**scores)
    Path(f'results/experiment{args.experiment}/normal_feasibility.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
