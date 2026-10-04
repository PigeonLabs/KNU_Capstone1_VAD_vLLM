"""Descriptive fixed-threshold event coverage; misses stay explicit."""
import numpy as np


def anomaly_events(labels, alarm):
    labels=np.asarray(labels);alarm=np.asarray(alarm,dtype=bool)
    if labels.ndim!=1 or alarm.shape!=labels.shape:raise ValueError('Label/alarm shape mismatch')
    edges=np.diff(np.r_[False,labels==1,False].astype(int));events=[]
    for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        hits=np.flatnonzero(alarm[start:end])
        events.append({'start_frame':int(start),'end_frame_exclusive':int(end),'length_frames':int(end-start),
                       'detected':bool(len(hits)),'first_alarm_delay_frames':int(hits[0]) if len(hits) else None,
                       'alarmed_fraction':float(alarm[start:end].mean()),
                       'alarm_active_before_onset':bool(start>0 and alarm[start-1]),
                       'left_boundary_censored':bool(start==0 or labels[start-1]<0),
                       'right_boundary_censored':bool(end==len(labels) or labels[end]<0)})
    return events


def summarize_events(events, short_limit=12):
    detected=[e for e in events if e['detected']];short=[e for e in events if e['length_frames']<=short_limit]
    return {'events':len(events),'detected_events':len(detected),'missed_events':len(events)-len(detected),
            'event_coverage':len(detected)/len(events) if events else None,
            'median_delay_detected_only_frames':float(np.median([e['first_alarm_delay_frames'] for e in detected])) if detected else None,
            'short_event_limit_frames':short_limit,'short_events':len(short),'detected_short_events':sum(e['detected'] for e in short),
            'detected_with_alarm_already_active':sum(e['detected'] and e['alarm_active_before_onset'] and e['first_alarm_delay_frames']==0 for e in events),
            'left_boundary_censored_events':sum(e['left_boundary_censored'] for e in events)}
