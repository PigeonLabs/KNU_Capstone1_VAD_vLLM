"""Post-evaluation decomposition of q99 changes; does not choose or refit a model."""
import json
from pathlib import Path
import numpy as np
from ipad_vad.data import hold_scores
from ipad_vad.events import anomaly_events,summarize_events
from audit_bank_dispatch import AFTER,PAIRS
from audit_missing_age import load,STATE_NAMES,direct_routes


def main():
    out=Path('results/experiment26');result={}
    for e in AFTER:
        current=json.loads(Path(f'results/experiment{e}/metrics.json').read_text());previous=json.loads(Path(f'results/experiment{PAIRS[e]}/metrics.json').read_text());q=current['normal_q99_threshold'];oldq=previous['normal_q99_threshold'];rows={s:{'added_normal':0,'added_anomaly':0,'removed_normal':0,'removed_anomaly':0} for s in STATE_NAMES};own_events=[];fixed_events=[]
        for p in sorted(Path(f'artifacts/experiment{e}/predictions').glob('*.npz')):
            v=load(p);seq=p.stem.split('_')[1];d=load(f'artifacts/experiment26/features/R04/testing_{seq}.npz');state=hold_scores(d['indices'],direct_routes(d,'24_age',56)[1],len(v['labels']));own=v['combined']>q;fixed=v['combined']>oldq
            own_events.extend([dict(x,sequence=p.stem) for x in anomaly_events(v['labels'],own)]);fixed_events.extend([dict(x,sequence=p.stem) for x in anomaly_events(v['labels'],fixed)])
            for i,name in enumerate(STATE_NAMES):
                for y,label in [(0,'normal'),(1,'anomaly')]:
                    mask=(state==i)&(v['labels']==y);rows[name]['added_'+label]+=int(np.sum(mask&own&~fixed));rows[name]['removed_'+label]+=int(np.sum(mask&fixed&~own))
        result[e]={'own_normal_q99':q,'legacy_normal_q99':oldq,'own_minus_legacy_threshold_on_same_new_scores':rows,'totals':{k:sum(v[k] for v in rows.values()) for k in next(iter(rows.values()))},'own_q_events':summarize_events(own_events),'legacy_q_on_new_scores_events':summarize_events(fixed_events)}
    (out/'threshold_decomposition.json').write_text(json.dumps({'post_evaluation_diagnostic_only':True,'variants':result,'note':'Same saved new score arrays evaluated at own and paired legacy normal q99. No model, threshold or gate selected; main results retain own q99.'},indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
