"""Experiment16 normal-only feasibility check, before development test evaluation."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process


def main():
    cfg=json.loads(Path('configs/experiment16.json').read_text());scene=cfg['scene'];split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    root=Path(f'artifacts/experiment16/features/{scene}')
    def load(seq):
        with np.load(root/f'training_{seq}.npz',allow_pickle=False) as f:return dict(f)
    fit=[load(s) for s in split['fit']];cal=[load(s) for s in split['calibration']]
    with threadpool_limits(limits=4):
        model=LognormalDwellBaseline(cfg,load_process(cfg));model.fit(fit);model.calibrate(cal);results=[model.score(d) for d in cal]
    keys=['visual','process','combined','transition','dwell','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']
    scores={k:np.concatenate([r[k] for r in results]) for k in keys};scores['sequence']=np.concatenate([np.full(len(d['indices']),s) for s,d in zip(split['calibration'],cal)])
    with np.load('artifacts/experiment15/normal_calibration_scores.npz',allow_pickle=False) as old:
        for key in ['visual','transition','dwell_valid','dwell_age','dwell_reason','dwell_entry_context','sequence']:np.testing.assert_array_equal(old[key],scores[key])
        old_ones=old['combined']==1
    assert np.quantile(scores['combined'],cfg['calibration_quantile'],method='higher')==model.threshold
    finite=all(np.isfinite(scores[k]).all() for k in ['visual','transition','dwell','process','combined'])
    bounds=all(np.all((scores[k]>=0)&(scores[k]<=1)) for k in ['visual','transition','dwell','process','combined'])
    parameters={f'{a}->{b}':{'complete_runs':len(model.dwell.context_durations[(a,b)]),'mu':mu,'sigma':sigma,'fit_max_duration':float(model.dwell.context_durations[(a,b)].max())} for (a,b),(mu,sigma) in model.dwell.log_parameters.items()}
    rows=[{'sequence':seq,'samples':len(r['combined']),'dwell_at_one':int(np.sum(r['dwell']==1)),'combined_at_one':int(np.sum(r['combined']==1)),'sample_alarm_count':int(np.sum(r['combined']>model.threshold))} for seq,r in zip(split['calibration'],results)]
    result={'normal_only':True,'fit_sequences':split['fit'],'calibration_sequences':split['calibration'],'calibration_samples':len(scores['combined']),'parameters':parameters,'sigma_floor':model.dwell.sigma_floor,'all_branch_scores_finite':finite,'all_branch_scores_in_unit_interval':bounds,'normal_q99':model.threshold,'normal_q99_below_closed_upper_bound':bool(model.threshold<1),'alarm_not_structurally_blocked':bool(finite and bounds and model.threshold<1),'dwell_at_one':int(np.sum(scores['dwell']==1)),'combined_at_one':int(np.sum(scores['combined']==1)),'old_ceiling_samples':{'count':int(old_ones.sum()),'ages':scores['dwell_age'][old_ones].tolist(),'new_dwell_scores':scores['dwell'][old_ones].tolist(),'new_combined_scores':scores['combined'][old_ones].tolist()},'videos':rows,'limitations':['Necessary feasibility diagnostic, not a guarantee of useful test alarms.','Lognormal shape is a modelling assumption; no claim of independently verified distribution fit.','Normal validation does not change sigma floor, threshold rule or model choice.']}
    np.savez_compressed('artifacts/experiment16/preflight_calibration_scores.npz',**scores)
    Path('results/experiment16/normal_feasibility.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
