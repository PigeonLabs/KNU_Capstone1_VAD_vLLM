"""Report branch contributions and observed/held phase conditioning after experiment18."""
import json
from pathlib import Path
import numpy as np
from ipad_vad.data import hold_scores


def main():
    out=Path('results/experiment18');m=json.loads((out/'metrics.json').read_text());q=m['normal_q99_threshold'];split=json.loads(Path('results/stage00/splits.json').read_text())['R04']
    fit={}
    for seq in split['fit']:
        with np.load(f'artifacts/experiment18/features/R04/training_{seq}.npz') as d:
            for phase in range(4):
                row=fit.setdefault(str(phase),{'all_samples':0,'observed_samples':0,'unobserved_samples':0});mask=d['phases']==phase
                row['all_samples']+=int(mask.sum());row['observed_samples']+=int(np.sum(mask&d['relation_valid']));row['unobserved_samples']+=int(np.sum(mask&~d['relation_valid']))
    branch={k:{'normal':0,'anomaly':0} for k in ['visual','transition','dwell','process_only_over_visual']};strata={};dwell_max=age_max=0.
    for path in Path('artifacts/experiment18/predictions').glob('*.npz'):
        with np.load(path) as d,np.load(f'artifacts/experiment18/features/R04/testing_{path.stem.split("_")[1]}.npz') as f:
            valid=hold_scores(f['indices'],f['relation_valid'],len(d['labels']));alarm=d['combined']>q
            for value,name in [(0,'normal'),(1,'anomaly')]:
                for observed in [False,True]:
                    mask=(d['labels']==value)&(valid==observed);row=strata.setdefault(f'{name}_observed_{observed}',{'frames':0,'alarms':0});row['frames']+=int(mask.sum());row['alarms']+=int(np.sum(mask&alarm))
                for key in ['visual','transition','dwell']:branch[key][name]+=int(np.sum((d['labels']==value)&(d[key]>q)))
                branch['process_only_over_visual'][name]+=int(np.sum((d['labels']==value)&alarm&~(d['visual']>q)))
            if d['dwell_valid'].any():dwell_max=max(dwell_max,float(d['dwell'][d['dwell_valid']].max()));age_max=max(age_max,float(d['dwell_age'][d['dwell_valid']].max()))
    result={'normal_fit_phase_conditioning':fit,'test_observation_strata':strata,'branch_exceedances':branch,'max_valid_dwell_score':dwell_max,'max_valid_dwell_age':age_max,'q99':q,'interpretation':'Unobserved samples still receive held/initial phase and currently enter phase-specific appearance subspaces. Counts diagnose modelling assumptions, not phase ground-truth errors. Branch exceedances overlap.'}
    (out/'observation_conditioning.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
