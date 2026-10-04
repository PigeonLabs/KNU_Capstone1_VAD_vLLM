"""Post-evaluation phase-stratified errors; does not select model parameters."""
import argparse,json
from pathlib import Path
import numpy as np
from ipad_vad.data import hold_scores


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiments',nargs='+',required=True);args=p.parse_args()
    rows=[]
    for n in args.experiments:
        m=json.loads(Path(f'results/experiment{n}/metrics.json').read_text());buckets={}
        for path in sorted(Path(f'artifacts/experiment{n}/predictions').glob('*.npz')):
            with np.load(path) as d:
                phases=hold_scores(d['indices'],d['phases'],len(d['labels'])).astype(int)
                labels=d['labels'];alarm=d['combined']>m['normal_q99_threshold']
                for phase in np.unique(phases):
                    mask=phases==phase;r=buckets.setdefault(int(phase),[0,0,0,0])
                    r[0]+=int((mask&(labels==0)).sum());r[1]+=int((mask&(labels==0)&alarm).sum())
                    r[2]+=int((mask&(labels==1)).sum());r[3]+=int((mask&(labels==1)&alarm).sum())
        for phase,(normal,fp,anomaly,tp) in sorted(buckets.items()):
            rows.append({'experiment':n,'phase':phase,'normal_frames':normal,'false_alarm_frames':fp,'normal_false_alarm_rate':fp/normal if normal else None,
                         'anomaly_frames':anomaly,'detected_anomaly_frames':tp,'anomaly_recall':tp/anomaly if anomaly else None})
    root=Path('results/comparison'+'_'.join(args.experiments));root.mkdir(parents=True,exist_ok=True)
    result={'rows':rows,'note':'Post-evaluation descriptive diagnosis on proxy phases. Each model uses its own normal q99. No phase ground truth or causal attribution to threshold versus ranking.'}
    (root/'phase_errors.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
