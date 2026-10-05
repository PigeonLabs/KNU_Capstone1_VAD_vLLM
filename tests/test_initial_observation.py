import copy
import numpy as np
import pytest
from ipad_vad.initial_observation import initial_observation_route
from test_request_calibration import models
from test_observed_appearance import setup


def test_current_observation_enables_phase_and_later_missing_does_not_reset():
    m,_=setup('phase');m.cfg['appearance_initial_observation_gate']=True
    d={'phases':np.array([0,0,2,2,2,1]),'relation_valid':np.array([False,False,True,False,False,True])};before=copy.deepcopy(d)
    np.testing.assert_array_equal(m.appearance_phases(d),[-1,-1,2,2,2,1])
    np.testing.assert_array_equal(m.appearance_phases(d,'fit'),d['phases'])
    for k in d:np.testing.assert_array_equal(d[k],before[k])


def test_prefix_future_independence_no_observation_empty_and_video_reset():
    p=np.array([0,0,1,1,1,2]);v=np.array([False,False,True,False,True,True]);d={'phases':p,'relation_valid':v};full=initial_observation_route(d,p)
    for n in range(7):np.testing.assert_array_equal(initial_observation_route({k:x[:n] for k,x in d.items()},p[:n]),full[:n])
    other=copy.deepcopy(d);other['relation_valid'][3:]=False
    np.testing.assert_array_equal(initial_observation_route(other,p)[:3],full[:3])
    missing={'phases':p,'relation_valid':np.zeros(6,bool)};np.testing.assert_array_equal(initial_observation_route(missing,p),[-1]*6)
    np.testing.assert_array_equal(initial_observation_route(d,p),full)


def test_pool_and_age_are_noop_and_fit_cdf_banks_unchanged():
    ms,d=models();d=copy.deepcopy(d);d['relation_valid'][:3]=False
    for old in ms:
        old.cfg['appearance_calibration_dispatch']='actual_bank';old.fit([d]);old.calibrate([d]);new=copy.deepcopy(old);new.cfg['appearance_initial_observation_gate']=True;new.fit([d]);new.calibrate([d])
        for key in old.spaces:
            for attr in ['mean','basis']:np.testing.assert_array_equal(getattr(old.spaces[key],attr),getattr(new.spaces[key],attr))
        for key in old.request_calibration.references:np.testing.assert_array_equal(old.request_calibration.references[key],new.request_calibration.references[key])
        a,b=old.score(d),new.score(d)
        np.testing.assert_array_equal(a['visual'][3:],b['visual'][3:]);np.testing.assert_array_equal(a['process'],b['process'])
        if old.cfg['appearance_inference_conditioning']!='phase':
            np.testing.assert_array_equal(a['visual'],b['visual']);assert old.threshold==new.threshold


def test_unobserved_initial_id_does_not_change_pooled_visual_scores():
    ms,d=models();m=ms[0];m.cfg.update(appearance_calibration_dispatch='actual_bank',appearance_initial_observation_gate=True);d=copy.deepcopy(d);d['relation_valid'][:3]=False;m.fit([d]);m.calibrate([d]);before=m.score(d)['visual'];other=copy.deepcopy(d);other['phases'][:3]=1
    np.testing.assert_array_equal(m.appearance_phases(other)[:3],[-1]*3);np.testing.assert_array_equal(m.score(other)['visual'][:3],before[:3])
    assert (-1,0) not in m.spaces or (-1,1) in m.spaces


@pytest.mark.parametrize('value',[1,None,'yes'])
def test_invalid_gate_option(value):
    m,d=setup('phase');m.cfg['appearance_initial_observation_gate']=value
    with pytest.raises(ValueError,match='boolean'):m.appearance_phases(d)


def test_invalid_mask_and_shape_rejected():
    with pytest.raises(ValueError):initial_observation_route({'phases':np.arange(2),'relation_valid':np.array([0,1])},np.arange(2))
    with pytest.raises(ValueError):initial_observation_route({'phases':np.arange(2),'relation_valid':np.array([False,True])},np.arange(3))
