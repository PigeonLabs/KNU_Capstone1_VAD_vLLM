"""Normal complete-run durations grouped by observed predecessor for planning."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',default='11');args=p.parse_args();n=args.experiment
    cfg=json.loads(Path(f'configs/experiment{n}.json').read_text());scene=cfg['scene'];split=json.loads(Path('results/stage00/splits.json').read_text())[scene];groups={}
    for group in ('fit','calibration'):
        buckets={}
        for seq in split[group]:
            with np.load(f'artifacts/experiment{n}/features/{scene}/training_{seq}.npz') as d:
                phase=d['phases'];idx=d['indices'];valid=d['relation_valid'];starts=np.r_[0,np.flatnonzero(phase[1:]!=phase[:-1])+1];ends=np.r_[starts[1:],len(phase)]
                for start,end in zip(starts,ends):
                    if start==0 or end==len(phase) or not valid[start-1:end+1].all():continue
                    key=f'{int(phase[start-1])}->{int(phase[start])}'
                    buckets.setdefault(key,[]).append({'sequence':seq,'duration_frames':int(idx[end]-idx[start])})
        groups[group]={key:{'complete_runs':len(rows),'distinct_sequences':len({r['sequence'] for r in rows}),
            'duration_min_median_max':np.quantile([r['duration_frames'] for r in rows],[0,.5,1]).tolist(),'runs':rows} for key,rows in sorted(buckets.items())}
    result={'scene':scene,'source':'NORMAL FIT/calibration only; post-experiment diagnostic, no test access','groups':groups,
        'note':'Entry context is the previous estimated cluster, not action or direction GT. Sparse normal contexts are not evidence of a reliable duration model.'}
    Path(f'results/experiment{n}/normal_entry_context_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({g:{k:{kk:vv for kk,vv in v.items() if kk!='runs'} for k,v in r.items()} for g,r in groups.items()},indent=2))

if __name__=='__main__':main()
