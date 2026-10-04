"""Scene-specific tracked role anchors without R01's corridor or spatial phase assumptions."""
import numpy as np


def select_role_anchors(data, role):
    chosen=np.full(len(data['indices']),-1,dtype=int);prior=None
    for step in range(len(chosen)):
        candidates=np.flatnonzero((data['object_frames']==step)&(data['roles']==role))
        if not len(candidates):continue
        same=candidates[data['tracks'][candidates]==prior]
        pool=same if len(same) else candidates
        chosen[step]=pool[np.argmax(data['confidence'][pool])]
        prior=data['tracks'][chosen[step]]
    return chosen


def fit_motion_axis(caches,role,min_track_samples=3,min_displacement=.1):
    moving=[];track_count=0
    for d in caches:
        for track in np.unique(d['tracks'][d['roles']==role]):
            boxes=d['boxes'][(d['roles']==role)&(d['tracks']==track)]
            centers=(boxes[:,:2]+boxes[:,2:])/2
            if len(centers)>=min_track_samples and np.ptp(centers,axis=0).max()>=min_displacement:
                moving.append(centers);track_count+=1
    if not moving:raise ValueError('No supported normal moving tracks; no R01 axis fallback')
    centers=np.concatenate(moving);variances=np.var(centers,axis=0);axis=int(np.argmax(variances))
    return axis,{'axis':axis,'axis_name':['x','y'][axis],'moving_tracks':track_count,'moving_observations':len(centers),
                 'coordinate_variances':variances.tolist(),'min_track_samples':min_track_samples,'min_displacement':min_displacement,
                 'note':'Dominant screen-coordinate axis; no assumption of monotonic motion or semantic phase order.'}
