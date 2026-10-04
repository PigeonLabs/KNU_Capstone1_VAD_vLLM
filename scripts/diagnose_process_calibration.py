"""Inspect state offsets of the global process calibration using NORMAL videos only."""
import argparse,json
from pathlib import Path
import numpy as np
from ipad_vad.experiment import load_process
from ipad_vad.scoring import empirical_percentile


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    cfg=json.loads(args.config.read_text());process=load_process(cfg);n=cfg['experiment'];scene=cfg['scene'];k=len(process['phases'])
    split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    with np.load(f'artifacts/experiment{n}/normal_model.npz') as f:
        transition=f['transition'];reference=f['process_reference']
        state_references={i:f[f'process_reference_state_{i}'] for i in range(k)} if 'process_calibration' in cfg else {}
    allowed=np.eye(k,dtype=bool);ids={v['id']:i for i,v in enumerate(process['phases'])};order=[ids[v] for v in process['normal_order']]
    for a,b in zip(order[:-1],order[1:]):allowed[a,b]=True
    if process['cyclic']:allowed[order[-1],order[0]]=True
    buckets={i:{'raw':[],'percentile':[],'self':[],'applied':[]} for i in range(k)}
    for seq in split['calibration']:
        with np.load(f'artifacts/experiment{n}/features/{scene}/training_{seq}.npz') as d:
            a,b=d['phases'][:-1],d['phases'][1:];raw=-np.log(transition[a,b])+(~allowed[a,b]).astype(float);score=empirical_percentile(reference,raw)
            for phase,r in buckets.items():
                use=a==phase;r['raw'].extend(raw[use].tolist());r['percentile'].extend(score[use].tolist());r['self'].extend((a[use]==b[use]).tolist())
                ref=state_references.get(phase,reference)
                if len(ref)<cfg.get('process_calibration',{}).get('minimum_support',1):ref=reference
                r['applied'].extend(empirical_percentile(ref,raw[use]).tolist())
    rows=[{'previous_phase':phase,'normal_calibration_transitions':len(v['raw']),
           'self_transition_fraction':float(np.mean(v['self'])) if v['self'] else None,
           'median_raw_nll_plus_penalty':float(np.median(v['raw'])) if v['raw'] else None,
            'median_global_percentile':float(np.median(v['percentile'])) if v['percentile'] else None} for phase,v in buckets.items()]
    if state_references:
        for row in rows:
            v=buckets[row['previous_phase']]['applied']
            row['median_applied_percentile']=float(np.median(v)) if v else None
    result={'scene':scene,'source':f'experiment{n} held-out NORMAL calibration only','rows':rows,
            'note':'Normal calibration reference reused for this diagnostic; descriptive offsets are not independent generalization evidence. Global and (when configured) applied conditional percentiles; no test labels or parameter tuning.'}
    Path(f'results/experiment{n}/normal_process_calibration_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
