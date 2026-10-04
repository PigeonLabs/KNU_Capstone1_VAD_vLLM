"""Post-evaluation error strata by observed entry context; no parameter selection."""
import argparse,json
from pathlib import Path
import numpy as np
from ipad_vad.data import hold_scores


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',default='10');p.add_argument('--experiment',default='11');args=p.parse_args()
    cfg=json.loads(Path(f'configs/experiment{args.experiment}.json').read_text());scene=cfg['scene'];roots=[Path(f'artifacts/experiment{n}') for n in (args.baseline,args.experiment)]
    thresholds=[json.loads(Path(f'results/experiment{n}/metrics.json').read_text())['normal_q99_threshold'] for n in (args.baseline,args.experiment)];buckets={}
    for path in sorted((roots[1]/'predictions').glob('*.npz')):
        seq=path.stem.split('_')[1]
        with np.load(roots[1]/'features'/scene/f'testing_{seq}.npz') as d,np.load(path) as new,np.load(roots[0]/'predictions'/path.name) as old:
            phase=d['phases'];observed=d['relation_valid'];context=np.full(len(phase),-1);entry=-1
            for i in range(1,len(phase)):
                if not observed[i] or not observed[i-1]:entry=-1
                elif phase[i]!=phase[i-1]:entry=int(phase[i-1])
                context[i]=entry
            context=hold_scores(d['indices'],context,len(new['labels']));dense_phase=hold_scores(d['indices'],phase,len(new['labels']));added=(new['combined']>thresholds[1])&~(old['combined']>thresholds[0]);y=new['labels']
            for before,now in np.unique(np.column_stack([context,dense_phase]),axis=0):
                key=f'{before}->{now}';use=(context==before)&(dense_phase==now)&new['dwell_valid']
                if not use.any():continue
                row=buckets.setdefault(key,{'normal_frames':0,'anomaly_frames':0,'added_false_positives':0,'added_true_positives':0})
                for name,mask in [('normal_frames',y==0),('anomaly_frames',y==1),('added_false_positives',added&(y==0)),('added_true_positives',added&(y==1))]:row[name]+=int((use&mask).sum())
    out={'scene':scene,'contexts':dict(sorted(buckets.items())),'note':'Observed entry clusters, not action labels. Post-evaluation strata on development labels; not used to select thresholds or support criteria.'}
    Path(f'results/experiment{args.experiment}/dwell_entry_errors.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
