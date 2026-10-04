"""Post-evaluation diagnostics of mean fusion; never tunes a score or threshold."""
import argparse,json
from pathlib import Path
import numpy as np
from ipad_vad.data import hold_scores
from ipad_vad.scoring import empirical_percentile


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',default='09');args=p.parse_args();n=args.experiment
    cfg=json.loads(Path(f'configs/experiment{n}.json').read_text());scene=cfg['scene']
    root=Path(f'artifacts/experiment{n}');metrics=json.loads(Path(f'results/experiment{n}/metrics.json').read_text());threshold=metrics['normal_q99_threshold']
    with np.load(root/'normal_model.npz') as d:
        refs={i:d[f'process_reference_state_{i}'] for i in range(len(d['transition']))};transition=d['transition'];global_ref=d['process_reference']
    self_rows=[]
    for i,ref in refs.items():
        if len(ref)<cfg['process_calibration']['minimum_support']:ref=global_ref
        percentile=float(empirical_percentile(ref,-np.log(transition[i,i])))
        cap=cfg['visual_process_weight']+(1-cfg['visual_process_weight'])*percentile
        self_rows.append({'state':i,'self_process_percentile':percentile,'combined_upper_bound_when_visual_is_one':cap,'can_exceed_current_q99':cap>threshold})
    buckets={name:{'frames':0,'normal':0,'anomaly':0,'false_positive':0,'true_positive':0,'combined_below_visual':0} for name in ['initial','self_transition','state_change']}
    for path in sorted((root/'predictions').glob('*.npz')):
        with np.load(path) as d:
            phase=d['phases'];kind=np.r_[0,np.where(phase[1:]==phase[:-1],1,2)]
            dense=hold_scores(d['indices'],kind,len(d['labels']));valid=d['labels']>=0;alarm=d['combined']>threshold
            for i,counts in enumerate(buckets.values()):
                use=(dense==i)&valid;y=d['labels'];counts['frames']+=int(use.sum());counts['normal']+=int((use&(y==0)).sum());counts['anomaly']+=int((use&(y==1)).sum())
                counts['false_positive']+=int((use&(y==0)&alarm).sum());counts['true_positive']+=int((use&(y==1)&alarm).sum())
                counts['combined_below_visual']+=int((use&(d['combined']<d['visual'])).sum())
    result={'scene':scene,'threshold':threshold,'self_transition_score_bounds':self_rows,'test_dense_strata':buckets,
            'interpretation':'Fixed arithmetic fusion bounds follow normal calibration only. Test labels describe strata after evaluation, not tune parameters. Self-transition means repeated estimated cluster, not a verified physical state.'}
    Path(f'results/experiment{n}/fusion_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
