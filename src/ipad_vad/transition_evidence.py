"""Causal observation evidence shared by transition calibration and fusion."""
import numpy as np


def observed_transition_mask(data):
    phases=np.asarray(data['phases']);valid=np.asarray(data['relation_valid'])
    if phases.ndim!=1 or valid.dtype!=np.bool_ or valid.shape!=phases.shape:
        raise ValueError('Boolean relation_valid must match one-dimensional phases')
    mask=np.zeros(len(phases),dtype=bool)
    mask[1:]=valid[:-1]&valid[1:]
    return mask


def same_track_pair_transition_mask(data):
    """Require two observed relations belonging to the same selected track pair.

    Detection indices are local to the video, not track IDs. Validate their sample
    alignment before gathering IDs; missing -1 entries must never index a track.
    """
    observed=observed_transition_mask(data);n=len(observed)
    selected=np.asarray(data['relation_detection_indices'])
    tracks=np.asarray(data['tracks']);frames=np.asarray(data['object_frames'])
    if selected.shape!=(n,2) or not np.issubdtype(selected.dtype,np.integer):
        raise ValueError('Two integer selected detection indices required per sample')
    if tracks.ndim!=1 or not np.issubdtype(tracks.dtype,np.integer):
        raise ValueError('One-dimensional integer track IDs required')
    if frames.shape!=tracks.shape or not np.issubdtype(frames.dtype,np.integer):
        raise ValueError('Integer detection frames must align with track IDs')
    if np.any((frames<0)|(frames>=n)):
        raise ValueError('Detection frame outside sample range')
    if np.any((selected < -1)|(selected>=len(tracks))):
        raise ValueError('Selected detection index outside cache')
    present=selected>=0
    if np.any(np.asarray(data['relation_valid'])&~present.all(axis=1)):
        raise ValueError('Observed relation requires both selected detections')
    pair=np.full((n,2),-1,dtype=tracks.dtype)
    expected_frames=np.broadcast_to(np.arange(n)[:,None],(n,2))
    if np.any(frames[selected[present]]!=expected_frames[present]):
        raise ValueError('Selected detection does not belong to its sample')
    if np.any(tracks[selected[present]]<0):
        raise ValueError('Selected track IDs must be nonnegative')
    pair[present]=tracks[selected[present]]
    same=np.zeros(n,dtype=bool);same[1:]=np.all(pair[:-1]==pair[1:],axis=1)
    return observed&same
