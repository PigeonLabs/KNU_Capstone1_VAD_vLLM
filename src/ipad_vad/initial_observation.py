"""Pooled appearance requests until a relation phase has actually been observed."""
import numpy as np


def initial_observation_route(data, requested):
    phases=np.asarray(data['phases']);valid=np.asarray(data['relation_valid']);requested=np.asarray(requested)
    if phases.ndim!=1 or valid.dtype!=np.bool_ or valid.shape!=phases.shape or requested.shape!=phases.shape:
        raise ValueError('Aligned one-dimensional phases, requests and boolean relation mask required')
    seen=np.logical_or.accumulate(valid)
    return np.where(seen,requested,-1)
