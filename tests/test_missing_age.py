import numpy as np
import pytest
from ipad_vad.missing_age import MissingAge,causal_age
from test_observed_appearance import setup


def sample(indices,valid,phases=None,seq='FIT/a'):
    return {'indices':np.array(indices),'relation_valid':np.array(valid,bool),'phases':np.array(phases if phases is not None else [1]*len(indices)),'sequence_id':seq}


def test_complete_gap_lengths_and_censor_exclusion_with_irregular_indices():
    d=sample([0,4,8,12,20,24,28,32,36,40,44],[0,1,0,0,1,0,1,0,0,1,0]);m=MissingAge(.9);m.fit([d])
    np.testing.assert_array_equal(m.durations,[12,4,8]);assert m.tau==12
    assert len(m.censored)==2 and m.censored[0]['left_censored'] and m.censored[1]['right_censored']
    assert m.censored[0]['observed_span_lower_bound_frames']==0


def test_exact_boundary_initial_missing_reset_and_prefix_invariance():
    m=MissingAge();m.fit([sample([0,4,8,12],[1,0,0,1])]);assert m.tau==8
    d=sample([0,4,8,12,16,20,24,28],[0,1,0,0,0,1,0,0],[9,2,8,8,8,3,9,9])
    age,state,phases=m.states(d)
    np.testing.assert_array_equal(age,[-1,0,4,8,12,0,4,8]);np.testing.assert_array_equal(state,[1,0,2,2,3,0,2,2]);np.testing.assert_array_equal(phases,[-1,2,2,2,-1,3,3,3])
    for end in range(1,len(d['indices'])+1):
        prefix={k:v[:end] if isinstance(v,np.ndarray) else v for k,v in d.items()}
        for actual,expected in zip(m.states(prefix),(age,state,phases)):np.testing.assert_array_equal(actual,expected[:end])
    # Reusing the model on a new sequence does not carry its last observation.
    np.testing.assert_array_equal(m.states(sample([0,4],[0,0]))[2],[-1,-1])


def test_no_complete_gaps_and_duplicate_ids_fail():
    m=MissingAge();d=sample([0,4,8],[0,1,0])
    with pytest.raises(ValueError,match='No complete'):m.fit([d])
    with pytest.raises(ValueError,match='distinct'):m.fit([d,d])
    with pytest.raises(RuntimeError):m.states(d)


@pytest.mark.parametrize('q',[0,-1,1.1,True,float('nan')])
def test_invalid_quantile(q):
    with pytest.raises(ValueError):MissingAge(q)


def test_invalid_time_or_mask():
    for idx in [[4,0],[0,0],[0.,4.],[-4,0]]:
        with pytest.raises(ValueError):causal_age(sample(idx,[1,0]))
    d=sample([0,4],[1,0]);d['relation_valid']=np.array([1,0])
    with pytest.raises(ValueError):causal_age(d)


def test_model_same_fit_banks_calibration_route_and_unsupported_fallback():
    m,d=setup();base,_=setup();d['sequence_id']='FIT/a';m.cfg['appearance_fit_conditioning']='observed_relation';m.cfg['appearance_inference_conditioning']='missing_age';m.cfg['appearance_missing_age']={'quantile':.9}
    for model in [base,m]:model.fit([d])
    for key in m.spaces:
        for attr in ['mean','basis']:np.testing.assert_array_equal(getattr(base.spaces[key],attr),getattr(m.spaces[key],attr))
    # Phase 9 is observed but unsupported, and must still use the ordinary pooled fallback.
    test={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in d.items()};test['phases'][0]=9
    assert m.appearance_space_key(-1,m.appearance_phases(test)[0])==(-1,-1)
    m.calibrate([d]);raw=m.raw(d)[0]
    for role,_,r in raw:np.testing.assert_array_equal(m.calibration[role],r)
    np.testing.assert_array_equal(m.raw(d)[1],base.raw(d)[1])
