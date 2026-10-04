import numpy as np
import pytest
from ipad_vad.lognormal_dwell import LognormalContextDwell,LognormalDwellBaseline
from ipad_vad.completed_dwell import CompletedContextDwell,CompletedDwellBaseline
from test_dwell import data
from test_baseline import fixture
from test_process_calibration import model


def test_known_lognormal_cdf_support_and_causality():
    training=[data([0]+[2]*(length//4)+[0]) for length in [4,16]]
    m=LognormalContextDwell(minimum_complete_runs=2);m.fit(training);m.calibrate([])
    probe=data([0]+[2]*12+[0]);score,valid,age,reason=m.score(probe)
    assert score[np.flatnonzero(age==0)[-1]]==0
    assert score[np.flatnonzero(age==8)[0]]==pytest.approx(.5)
    assert score[np.flatnonzero(age==16)[0]]==pytest.approx(.8413447460685429)
    assert np.all(score[valid&(age>16)]<1)
    before=score.copy();m.calibrate([data([0]+[2]*500+[0])]);np.testing.assert_array_equal(before,m.score(probe)[0])
    changed={k:v.copy() for k,v in probe.items()};changed['phases'][6:]=3
    np.testing.assert_array_equal(before[:6],m.score(changed)[0][:6])
    old=CompletedContextDwell(minimum_complete_runs=2);old.fit(training);old.calibrate([])
    for a,b in zip((valid,age,reason),old.score(probe)[1:]):np.testing.assert_array_equal(a,b)
    assert len(m.reference)==0


def test_constant_lengths_use_sigma_floor_and_expose_numeric_saturation():
    m=LognormalContextDwell(minimum_complete_runs=2);m.fit([data([0,2,2,0])]*2);m.calibrate([])
    assert m.log_parameters[(0,2)][1]==.05
    score,valid,age,_=m.score(data([0]+[2]*10+[0]))
    assert np.isfinite(score).all() and np.all((score>=0)&(score<=1))
    assert score[np.flatnonzero(age==8)[0]]==.5
    assert np.any(score[valid]==1) # Do not conceal finite-precision saturation with clipping.
    with pytest.raises(ValueError):LognormalContextDwell(sigma_floor=0)


def test_full_model_preserves_non_dwell_branches_and_normal_q99():
    template=model();cfg=dict(template.cfg,score_fusion='max',normal_dwell={'minimum_complete_runs':10},dwell_context={'distribution':'entry'},dwell_score='fit_lognormal_cdf',dwell_lognormal={'sigma_floor':.05})
    def f(seed):
        d=fixture(seed);d['relation_valid']=np.ones(len(d['phases']),bool);return d
    old=CompletedDwellBaseline(cfg,template.process);new=LognormalDwellBaseline(cfg,template.process)
    for m in (old,new):m.fit([f(0),f(1)]);m.calibrate([f(2)])
    a=old.score(f(3));b=new.score(f(3))
    for key in ['visual','transition','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']:np.testing.assert_array_equal(a[key],b[key])
    assert new.threshold==np.quantile(new.score(f(2))['combined'],.99,method='higher')
