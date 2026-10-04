"""Post-evaluation error description; never chooses model parameters."""
import argparse,json
from pathlib import Path
import numpy as np
from ipad_vad.data import hold_scores


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',default='04');p.add_argument('--baseline',default='03');args=p.parse_args()
    m=json.loads(Path(f'results/experiment{args.experiment}/metrics.json').read_text())
    rows=[];runs_all={'false_alarm':[],'true_alarm':[]};values={'test_normal_valid':[],'test_anomaly_valid':[]}
    for path in sorted(Path(f'artifacts/experiment{args.experiment}/predictions').glob('*.npz')):
        with np.load(path) as f:prediction=dict(f)
        with np.load(Path(f'artifacts/experiment{args.baseline}/predictions')/path.name) as f:old=dict(f)
        assert np.array_equal(prediction['visual'],old['visual'])
        assert np.array_equal(prediction['transition'],old.get('transition',old['process']))
        scene,seq=path.stem.split('_')
        with np.load(f'artifacts/experiment{args.experiment}/features/{scene}/testing_{seq}.npz') as f:d=dict(f)
        y=prediction['labels'];valid=prediction['motion_valid'];alarm=prediction['combined']>m['normal_q99_threshold']
        velocity=hold_scores(d['indices'],d['motion_velocity'],len(y))
        for label,name in [(0,'test_normal_valid'),(1,'test_anomaly_valid')]:values[name].extend(velocity[(y==label)&valid].tolist())
        row={'sequence':seq,'normal_frames':int((y==0).sum()),'false_alarm_frames':int((alarm&(y==0)).sum()),
             'anomaly_frames':int((y==1).sum()),'detected_anomaly_frames':int((alarm&(y==1)).sum())}
        for label,name in [(0,'false_alarm'),(1,'true_alarm')]:
            edges=np.diff(np.r_[False,alarm&(y==label),False].astype(int))
            runs=(np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)).tolist()
            runs_all[name].extend(runs);row[name+'_runs_source_frames']=runs
        rows.append(row)
    result={'threshold':m['normal_q99_threshold'],'unchanged_visual_and_transition_predictions':True,
        'scope':'post-evaluation error diagnosis; not used to tune this experiment',
        'velocity_quantiles':{k:np.quantile(x,[.01,.25,.5,.75,.99]).tolist() if x else None for k,x in values.items()},
        'alarm_run_summary':{k:{'runs':len(x),'median_frames':float(np.median(x)) if x else None,'max_frames':max(x,default=0),
             'fraction_runs_at_most_4_frames':float(np.mean(np.array(x)<=4)) if x else None} for k,x in runs_all.items()},'sequences':rows}
    Path(f'results/experiment{args.experiment}/error_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['alarm_run_summary'],indent=2))

if __name__=='__main__':main()
