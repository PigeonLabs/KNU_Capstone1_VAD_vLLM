"""Count-matched normal FIT controls; relation masks determine counts only."""
import numpy as np


def sampling_rng(config):
    spec=config.get('appearance_fit_sampling')
    if spec is None:return None
    mode=config.get('appearance_fit_conditioning',config.get('appearance_conditioning','phase'))
    if mode!='phase':raise ValueError('Random FIT sampling requires the full phase population')
    if not isinstance(spec,dict) or set(spec)!={'mode','seed'} or spec['mode']!='count_matched_random':
        raise ValueError('Unknown appearance FIT sampling specification')
    if type(spec['seed']) is not int or spec['seed']<0:raise ValueError('Sampling seed must be a nonnegative integer')
    return np.random.default_rng(spec['seed'])


def observed_mask(data):
    valid=np.asarray(data['relation_valid'])
    if valid.dtype!=np.bool_ or valid.shape!=data['phases'].shape:
        raise ValueError('relation_valid must be a boolean array matching phases')
    return valid
