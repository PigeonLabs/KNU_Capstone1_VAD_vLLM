"""Post-evaluation branch and lost-event diagnosis without model selection."""
import json
from pathlib import Path
import numpy as np
from ipad_vad.data import hold_scores


def main():
    out=Path('results/experiment19');q=json.loads((out/'metrics.json').read_text())['normal_q99_threshold'];counts={};max_dwell=0.
    for path in Path('artifacts/experiment19/predictions').glob('*.npz'):
        with np.load(path) as d:
            masks={k:d[k]>q for k in ['visual','transition','dwell']};masks['process_only_over_visual']=(d['combined']>q)&~masks['visual']
            for name,mask in masks.items():
                row=counts.setdefault(name,{'normal':0,'anomaly':0})
                for value,label in [(0,'normal'),(1,'anomaly')]:row[label]+=int(np.sum(mask&(d['labels']==value)))
            max_dwell=max(max_dwell,float(d['dwell'].max()))
    events=json.loads(Path('results/comparison18_19/events.json').read_text());lost=[]
    for event in events['lost_events']:
        row=dict(event);key=row['sequence_key'];seq=key.split('_')[1]
        with np.load(f'artifacts/experiment18/features/R04/testing_{seq}.npz') as f,np.load(f'artifacts/experiment18/predictions/{key}.npz') as a,np.load(f'artifacts/experiment19/predictions/{key}.npz') as b:
            valid=hold_scores(f['indices'],f['relation_valid'],len(a['labels'])).astype(bool);interval=np.zeros(len(valid),bool);interval[row['start_frame']:row['end_frame_exclusive']]=True
            row.update(frames=int(interval.sum()),observed_frames=int(np.sum(interval&valid)),old_alarm_observed=int(np.sum(interval&valid&(a['combined']>q))),old_alarm_unobserved=int(np.sum(interval&~valid&(a['combined']>q))),new_alarms=int(np.sum(interval&(b['combined']>q))))
        lost.append(row)
    result={'q99':q,'branch_exceedances':counts,'max_dwell_score':max_dwell,'lost_event_observation_diagnostic':lost,'note':'Counts overlap between branches. Fixed q99 happens to match both experiments. Post-evaluation descriptive diagnosis, no tuning.'}
    (out/'branch_alarms.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
