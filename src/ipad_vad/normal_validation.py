"""Normal-video holdout calibration with immutable fit-model evidence."""
from copy import deepcopy
import hashlib
import numpy as np
from ipad_vad.process_calibration import StateCalibratedBaseline
from ipad_vad.dwell_scoring import DwellBaseline
from ipad_vad.context_dwell_scoring import ContextDwellBaseline


def leave_one_video_out(fit_ids,calibration_ids):
    if len(set(fit_ids))!=len(fit_ids) or len(set(calibration_ids))!=len(calibration_ids):raise ValueError('Duplicate video IDs')
    if set(fit_ids)&set(calibration_ids):raise ValueError('Fit and calibration pool overlap')
    if len(calibration_ids)<2:raise ValueError('At least two calibration videos required')
    return [{'held_out':held,'calibration':[s for s in calibration_ids if s!=held]} for held in calibration_ids]


def make_model(config,process):
    if config.get('process_calibration',{}).get('mode')!='previous_state' or 'normal_progress' in config:raise ValueError('Unsupported normal-validation model')
    if 'dwell_context' in config:return ContextDwellBaseline(config,process)
    if 'normal_dwell' in config:return DwellBaseline(config,process)
    return StateCalibratedBaseline(config,process)


def fit_arrays(model):
    result={'transition':model.transition,'allowed':model.allowed}
    for (role,phase),space in model.spaces.items():result[f'mean_{role}_{phase}']=space.mean;result[f'basis_{role}_{phase}']=space.basis
    if hasattr(model,'dwell'):
        for state,values in model.dwell.durations.items():result[f'dwell_durations_state_{state}']=values
        if hasattr(model.dwell,'context_durations'):
            for (a,b),values in model.dwell.context_durations.items():result[f'dwell_context_durations_{a}_{b}']=values
    return result


def array_fingerprint(arrays):
    digest=hashlib.sha256()
    for key,value in sorted(arrays.items()):
        x=np.ascontiguousarray(value);digest.update(key.encode());digest.update(str(x.dtype).encode());digest.update(str(x.shape).encode());digest.update(x.tobytes())
    return digest.hexdigest()


def reference_arrays(model):
    result={'process_reference':model.process_reference,'threshold':np.array(model.threshold)}
    for role,values in model.calibration.items():result[f'calibration_{role}']=values
    for state,values in model.state_process_references.items():result[f'process_reference_state_{state}']=values
    if hasattr(model,'dwell'):result['dwell_reference']=model.dwell.reference
    return result


def calibrate_holdout(fitted_model,caches,calibration_ids,held_out):
    if held_out in calibration_ids or len(set(calibration_ids))!=len(calibration_ids):raise ValueError('Holdout must be excluded and calibration IDs unique')
    before=array_fingerprint(fit_arrays(fitted_model));model=deepcopy(fitted_model)
    model.calibrate([caches[s] for s in calibration_ids])
    cal_scores=[model.score(caches[s]) for s in calibration_ids]
    held_scores=model.score(caches[held_out])
    if array_fingerprint(fit_arrays(model))!=before:raise AssertionError('Calibration changed the fitted model')
    return model,cal_scores,held_scores
