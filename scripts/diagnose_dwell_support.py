"""Normal FIT cluster-run support for planning a duration model; no test access."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',default='10');args=parser.parse_args();n=args.experiment
    cfg=json.loads(Path(f'configs/experiment{n}.json').read_text());scene=cfg['scene'];split=json.loads(Path('results/stage00/splits.json').read_text())[scene];rows=[]
    for seq in split['fit']:
        with np.load(f'artifacts/experiment{n}/features/{scene}/training_{seq}.npz') as d:
            phase=d['phases'];idx=d['indices'];valid=d['relation_valid'];starts=np.r_[0,np.flatnonzero(phase[1:]!=phase[:-1])+1];ends=np.r_[starts[1:],len(phase)]
            for start,end in zip(starts,ends):
                rows.append({'sequence':seq,'state':int(phase[start]),'left_boundary':bool(start==0),'right_boundary':bool(end==len(phase)),
                    'all_relations_valid':bool(valid[start:end].all()),'duration_frames':int(idx[end]-idx[start]) if end<len(phase) else int(d['frame_count']-idx[start])})
    summary=[]
    for state in range(cfg['relational_phase']['k']):
        r=[r for r in rows if r['state']==state];complete=[r['duration_frames'] for r in r if not r['left_boundary'] and not r['right_boundary'] and r['all_relations_valid']]
        summary.append({'state':state,'all_runs':len(r),'left_boundary_runs':sum(x['left_boundary'] for x in r),'right_boundary_runs':sum(x['right_boundary'] for x in r),
            'invalid_relation_runs':sum(not x['all_relations_valid'] for x in r),'complete_internal_valid_runs':len(complete),
            'complete_duration_min_median_max':None if not complete else [min(complete),float(np.median(complete)),max(complete)]})
    out={'source':scene+' NORMAL FIT only; post-experiment planning diagnostic','units':'source_frame_index','states':summary,'runs':rows,
        'note':'Cluster runs, not verified action durations. Sampling resolution 4 frames. Initial/terminal runs are boundary-censored; observed missingness can invalidate a run. No test labels used.'}
    Path(f'results/experiment{n}/normal_dwell_support.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
