"""Causal observation evidence shared by transition calibration and fusion."""
import numpy as np


def observed_transition_mask(data):
    phases=np.asarray(data['phases']);valid=np.asarray(data['relation_valid'])
    if phases.ndim!=1 or valid.dtype!=np.bool_ or valid.shape!=phases.shape:
        raise ValueError('Boolean relation_valid must match one-dimensional phases')
    mask=np.zeros(len(phases),dtype=bool)
    mask[1:]=valid[:-1]&valid[1:]
    return mask


