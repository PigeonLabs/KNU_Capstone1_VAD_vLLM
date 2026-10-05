"""Causal identity continuity from an observed phase entry to the present."""
import numpy as np
from ipad_vad.transition_evidence import same_track_pair_transition_mask


def same_pair_since_entry_mask(data):
    """Missing observations or either changed track invalidate the entire history.

    A later same-pair edge alone cannot restore it: an observed phase change is
    required. No future exit or assumed entry at the start of a video is used.
    """
    same = same_track_pair_transition_mask(data)
    phases = np.asarray(data['phases'])
    mask = np.zeros(len(phases), dtype=bool)
    known = False
    for i in range(1, len(phases)):
        if not same[i]:
            known = False
        elif phases[i] != phases[i - 1]:
            known = True
        mask[i] = known
    return mask
