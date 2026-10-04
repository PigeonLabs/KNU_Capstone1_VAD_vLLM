"""Identity-bounded sampled-phase episodes with explicit incomplete observations.

This module creates duration evidence, not anomaly scores. Track IDs and latent
phases are observations, not semantic ground truth. Interrupted episodes provide
only the age at their last actual observation; gaps are never filled as exposure.
"""
import numpy as np
from ipad_vad.transition_evidence import same_track_pair_transition_mask


def extract_dwell_episodes(data):
    phases=np.asarray(data['phases']);indices=np.asarray(data['indices'])
    if phases.ndim!=1 or not np.issubdtype(phases.dtype,np.integer) or np.any(phases<0):
        raise ValueError('Nonnegative one-dimensional integer phases required')
    if indices.shape!=phases.shape or not np.issubdtype(indices.dtype,np.integer):
        raise ValueError('Integer source indices must match phases')
    if np.any(indices<0) or np.any(indices[1:]<=indices[:-1]):
        raise ValueError('Source indices must be nonnegative and strictly increasing')
    if 'frame_count' in data:
        count=np.asarray(data['frame_count'])
        if count.ndim or not np.issubdtype(count.dtype,np.integer) or int(count)<0 or (len(indices) and indices[-1]>=int(count)):
            raise ValueError('Frame count must bound all source indices')
    same=same_track_pair_transition_mask(data);valid=np.asarray(data['relation_valid'])
    selected=np.asarray(data['relation_detection_indices']);tracks=np.asarray(data['tracks'])
    n=len(phases);assignment=np.full(n,-1,dtype=np.int64);age=np.full(n,-1,dtype=np.int64);context=np.full(n,-1,dtype=np.int64)
    episodes=[];active=None

    def finish(end,reason):
        nonlocal active
        start=active['start_sample'];last=end-1;known=active['entry_observed'];exit_observed=reason=='phase_change'
        row=dict(active,end_sample_exclusive=end,last_observed_sample=last,
                 last_observed_source_frame=int(indices[last]),observed_samples=end-start,
                 observation_span_frames=int(indices[last]-indices[start]),
                 boundary_sample=end if end<n else None,boundary_source_frame=int(indices[end]) if end<n else None,
                 end_reason=reason,exit_observed=exit_observed,
                 status='complete' if known and exit_observed else 'right_censored' if known else 'unknown_entry',
                 duration_frames=int(indices[end]-indices[start]) if known and exit_observed else None,
                 censor_lower_bound_frames=int(indices[last]-indices[start]) if known and not exit_observed else None)
        episodes.append(row);active=None

    for i in range(n):
        if not valid[i]:
            if active is not None:finish(i,'relation_missing')
            continue
        pair=[int(x) for x in tracks[selected[i]]]
        reason=None;known=False;previous=None
        if active is None:reason='video_start' if i==0 else 'reacquired'
        elif not same[i]:
            anchor=pair[0]!=active['track_pair'][0];target=pair[1]!=active['track_pair'][1]
            reason='both_changed' if anchor and target else 'anchor_changed' if anchor else 'target_changed'
            finish(i,reason)
        elif phases[i]!=phases[i-1]:
            finish(i,'phase_change');reason='phase_change';known=True;previous=int(phases[i-1])
        if reason is not None:
            active={'episode_id':len(episodes),'start_sample':i,'first_observed_source_frame':int(indices[i]),
                    'phase':int(phases[i]),'track_pair':pair,'start_reason':reason,'entry_observed':known,
                    'entry_context':previous,'entry_source_frame':int(indices[i]) if known else None}
        assignment[i]=active['episode_id']
        if active['entry_observed']:
            age[i]=int(indices[i]-active['entry_source_frame']);context[i]=active['entry_context']
    if active is not None:finish(n,'video_end')
    return {'episodes':episodes,'sample_episode_ids':assignment,'sample_age_frames':age,'sample_entry_context':context,
            'missing_sample_indices':np.flatnonzero(~valid)}
