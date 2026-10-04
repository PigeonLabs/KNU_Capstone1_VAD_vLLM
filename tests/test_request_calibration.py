import copy
import numpy as np
import pytest
from ipad_vad.scoring import empirical_percentile
from ipad_vad.request_calibration import RequestCalibration,request_residuals
from test_observed_appearance import setup


def models():
    result=[];_,d=setup();d['sequence_id']='FIT/a'
    for mode in ['phase','observed_relation','missing_age']:
        m,_=setup();m.cfg.update(appearance_fit_conditioning='observed_relation',appearance_inference_conditioning=mode,appearance_request_calibration='dual_full_normal')
        if mode=='missing_age':m.cfg['appearance_missing_age']={'quantile':.9}
        m.fit([d]);m.calibrate([d]);result.append(m)
    return result,d


def test_same_full_population_references_and_score_invariance():
    ms,d=models()
    for m in ms:
        for role,frames,phase,pool in request_residuals(m,d):
            for req,x in [(1,phase),(0,pool)]:
                assert len(x)==40
                np.testing.assert_array_equal(m.request_calibration.references[role,req],x)
                np.testing.assert_array_equal(x,ms[0].request_calibration.references[role,req])
            raw=next(v for r,_,v in m.raw(d)[0] if r==role);req=m.appearance_phases(d)[frames]>=0
            expected=np.where(req,empirical_percentile(phase,phase),empirical_percentile(pool,pool))
            np.testing.assert_array_equal(m.calibrate_appearance(d,role,frames,raw),expected)
        np.testing.assert_array_equal(m.score(d)['process'],ms[0].score(d)['process'])
    for role in [-1,0]:
        outputs=[next(v for r,_,v in m.score(d)['objects'] if r==role) for m in ms]
        for v in outputs:np.testing.assert_array_equal(v[d['relation_valid']],outputs[0][d['relation_valid']])


def test_support_fallback_keeps_phase_request_reference_identity():
    ms,d=models();d=copy.deepcopy(d);d['relation_valid'][:20]=False
    for m in ms[:2]:m.fit([d]);m.calibrate([d])
    hold,pool=ms[:2];assert (-1,0) not in hold.spaces
    np.testing.assert_array_equal(hold.raw(d)[0][0][2][:20],pool.raw(d)[0][0][2][:20])
    assert np.all(hold.appearance_phases(d)[:20]>=0) and np.all(pool.appearance_phases(d)[:20]<0)
    for m in [hold,pool]:assert set(m.request_calibration.references)=={(-1,0),(-1,1),(0,0),(0,1)}
    assert not np.array_equal(hold.request_calibration.references[-1,0],hold.request_calibration.references[-1,1])


def test_excluded_video_never_enters_either_reference_and_count_order():
    ms,d=models();m=ms[0];other=copy.deepcopy(d);other['sequence_id']='CAL/b';other['global_features']+=10
    m.calibrate([other])
    assert all(v['videos']==['CAL/b'] and v['observations']==40 for v in m.request_calibration.support.values())
    for role,_,phase,pool in request_residuals(m,other):
        np.testing.assert_array_equal(m.request_calibration.references[role,1],phase);np.testing.assert_array_equal(m.request_calibration.references[role,0],pool)


def test_invalid_requests_and_config():
    ms,d=models();m=ms[0]
    with pytest.raises(ValueError):m.request_calibration.fit(m,[d,d])
    with pytest.raises(ValueError):m.request_calibration.score(-1,[2],[.1])
    with pytest.raises(ValueError):m.request_calibration.score(99,[0],[.1])
    m.cfg['appearance_request_calibration']='bad'
    with pytest.raises(ValueError):m.calibrate([d])
    m.cfg['appearance_request_calibration']='dual_full_normal';m.cfg['appearance_route_calibration']={}
    with pytest.raises(ValueError):m.calibrate([d])
