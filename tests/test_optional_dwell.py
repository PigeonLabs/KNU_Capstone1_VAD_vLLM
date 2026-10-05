import numpy as np
import pytest
from ipad_vad.optional_dwell import OptionalLognormalContextDwell
from ipad_vad.lognormal_dwell import LognormalContextDwell
from test_dwell import data


def test_no_completed_run_is_unavailable_not_a_fitted_normal_score():
    m=OptionalLognormalContextDwell(minimum_complete_runs=10);m.fit([data([3]*20)]);m.calibrate([])
    assert not m.available and m.support=={} and m.context_support=={} and not m.log_parameters and len(m.reference)==0
    score,valid,age,reason=m.score(data([0,1,1,2,2,3]))
    assert not valid.any() and np.all(score==0) and np.all(reason!=0)
    with pytest.raises(ValueError,match='No supported normal dwell state'):LognormalContextDwell().fit([data([3]*20)])


def test_pooled_support_does_not_replace_insufficient_context_support():
    m=OptionalLognormalContextDwell(minimum_complete_runs=2);m.fit([data([0,2,2,0]),data([1,2,2,1])]);m.calibrate([])
    assert m.support[2]==2 and len(m.durations[2])==2 and m.context_support=={'0->2':1,'1->2':1}
    assert not m.available and not m.context_durations and not m.score(data([0,2,2,0]))[1].any()


def test_supported_path_is_exact_to_strict_model_and_resets_after_refit():
    training=[data([0]+[2]*n+[0]) for n in [1,4,8]];probe=data([0]+[2]*15+[0])
    a=LognormalContextDwell(minimum_complete_runs=2);b=OptionalLognormalContextDwell(minimum_complete_runs=2)
    for m in [a,b]:m.fit(training);m.calibrate([])
    assert b.available and b.log_parameters==a.log_parameters
    for x,y in zip(a.score(probe),b.score(probe)):np.testing.assert_array_equal(x,y)
    b.fit([data([3]*20)]);b.calibrate([])
    assert not b.available and not b.score(probe)[1].any() and not b.log_parameters
