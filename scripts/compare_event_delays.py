"""Paired descriptive event coverage and delay; keep all misses and boundary caveats."""
import argparse,json
from pathlib import Path
import numpy as np
from ipad_vad.events import anomaly_events,summarize_events


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiments',nargs=2,required=True);args=p.parse_args()
    records={};summaries={};labels={}
    for n in args.experiments:
        m=json.loads(Path(f'results/experiment{n}/metrics.json').read_text());events=[];labels[n]={}
        for path in sorted(Path(f'artifacts/experiment{n}/predictions').glob('*.npz')):
            with np.load(path) as f:
                y=f['labels'];alarm=f['combined']>m['normal_q99_threshold']
                labels[n][path.stem]=y.copy()
            events.extend([dict(e,sequence_key=path.stem) for e in anomaly_events(y,alarm)])
        records[n]=events;summaries[n]=summarize_events(events)
    before,after=args.experiments
    assert labels[before].keys()==labels[after].keys()
    for seq,y in labels[before].items():assert np.array_equal(y,labels[after][seq]),seq
    matched=[];lost=[];gained=[];both_missed=[]
    for a,b in zip(records[before],records[after]):
        assert all(a[k]==b[k] for k in ['sequence_key','start_frame','end_frame_exclusive'])
        identity={k:a[k] for k in ['sequence_key','start_frame','end_frame_exclusive']}
        if a['detected'] and b['detected']:
            matched.append(dict(identity,delay_before=a['first_alarm_delay_frames'],delay_after=b['first_alarm_delay_frames'],
                                delay_change_frames=b['first_alarm_delay_frames']-a['first_alarm_delay_frames']))
        elif a['detected']:lost.append(identity)
        elif b['detected']:gained.append(identity)
        else:both_missed.append(identity)
    assert len(records[before])==len(records[after])
    result={'experiments':args.experiments,'time_unit':'source_frame_index','summaries':summaries,
            'paired_both_detected':matched,'lost_events':lost,'gained_events':gained,'both_missed':both_missed,
            'median_paired_delay_change_frames':float(np.median([e['delay_change_frames'] for e in matched])) if matched else None,
            'events':records,'limitations':['Each model uses its own normal q99; not equal test FPR.',
                'Event coverage means any alarm overlaps a contiguous positive GT interval. It is not point-adjusted frame AUROC.',
                'An alarm already active before onset can yield zero delay; flagged explicitly.',
                'Misses have null delay and remain in counts. Detected-only latency is conditional, not overall latency.',
                'Clip-boundary/unknown-label censoring flagged; FPS unavailable. Repeated scene-specific development data.']}
    root=Path('results/comparison'+'_'.join(args.experiments));root.mkdir(parents=True,exist_ok=True)
    (root/'events.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
