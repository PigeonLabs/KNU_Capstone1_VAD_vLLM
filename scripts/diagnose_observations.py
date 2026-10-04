"""Audit observation gaps without opening anomaly labels."""
import argparse,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.spatial_phase import SpatialPhase


def load(p):
    with np.load(p,allow_pickle=False) as f:return dict(f)


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);args=p.parse_args()
    cfg=json.loads(Path(f'configs/experiment{args.experiment}.json').read_text());scene=cfg['scene']
    split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    source=Path('artifacts/experiment01/features')/scene
    phase=SpatialPhase(**cfg['spatial_phase'])
    with threadpool_limits(limits=4):phase.fit([load(source/f'training_{s}.npz') for s in split['fit']])
    rows=[]
    for path in sorted(Path(f'artifacts/experiment{args.experiment}/features/{scene}').glob('*.npz')):
        d=load(path);_,observed,_,_=phase.transform(d)
        # Source-frame duration covered by each sampled observation, including final partial interval.
        widths=np.diff(np.r_[d['indices'],int(d['frame_count'])]);runs=[];current=0
        for valid,width in zip(observed,widths):
            if valid:
                if current:runs.append(int(current))
                current=0
            else:current+=width
        if current:runs.append(int(current))
        part,seq=path.stem.split('_');group='test' if part=='testing' else 'fit' if seq in split['fit'] else 'calibration'
        rows.append({'sequence_key':path.stem,'group':group,'samples':len(observed),'direct_observations':int(observed.sum()),
                     'observation_fraction':float(observed.mean()),'missing_run_lengths_source_frames':runs,
                     'maximum_missing_run_source_frames':max(runs,default=0),'product_boxes':int((d['roles']==0).sum())})
    aggregate={}
    for group in ('fit','calibration','test','all'):
        selected=[r for r in rows if group=='all' or r['group']==group]
        n=sum(r['samples'] for r in selected);obs=sum(r['direct_observations'] for r in selected)
        gaps=[v for r in selected for v in r['missing_run_lengths_source_frames']]
        aggregate[group]={'samples':n,'observations':obs,'observation_fraction':obs/n,'missing_run_count':len(gaps),
                          'max_gap_source_frames':max(gaps,default=0),'median_gap_source_frames':float(np.median(gaps)) if gaps else 0}
    out={'experiment':args.experiment,'phase_map_source':'experiment02 frozen, fitted only on original normal FIT tracks',
         'not_detection_recall':True,'time_unit':'source_frame_index','aggregate':aggregate,'sequences':rows}
    Path(f'results/experiment{args.experiment}/observations.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))

if __name__=='__main__':main()
