import copy
import numpy as np
import pytest
from ipad_vad.confirmed_anchor import ConfirmedAnchorPhase
from ipad_vad.semantic_priority_anchor import SemanticPriorityAnchorPhase
from test_semantic_priority_anchor import fixture,model


def direct(margins,eligible=None,tracks=(10,11,12)):
    n=len(margins);k=len(tracks)
    d=dict(selection_margins=np.array(margins,float).ravel(),tracks=np.tile(tracks,n),object_frames=np.repeat(np.arange(n),k),confidence=np.ones(n*k),anchor_confirmation_state={})
    m=model(ConfirmedAnchorPhase);prior=None;chosen=[]
    for step in range(n):
        ids=np.arange(step*k,(step+1)*k)
        if eligible is not None:ids=ids[np.array(eligible[step],bool)]
        i=m.select_candidate(d,ids,prior,1);chosen.append(None if i<0 else int(d['tracks'][i]))
        if i>=0:prior=int(d['tracks'][i])
    return chosen


def test_replacement_waits_for_same_challenger_and_ties_keep_prior():
    assert direct([[3,2,1],[1,3,2],[1,3,2],[3,3,1]])==[10,10,11,11]
    assert direct([[3,2,1],[1,3,2],[1,2,3],[1,2,3]])==[10,10,10,12]


def test_empty_and_forced_replacement_clear_pending_without_filling_missing():
    assert direct([[3,2,1],[1,3,2],[1,3,2],[1,3,2],[1,3,2]],[[1,1,1],[1,1,1],[0,0,0],[1,1,1],[1,1,1]])==[10,10,None,10,11]
    assert direct([[3,2,1],[1,3,2],[1,2,3],[1,3,2],[1,3,2]],[[1,1,1],[1,1,1],[0,1,1],[1,1,1],[1,1,1]])==[10,10,12,12,11]


def test_prior_recovery_clears_pending_and_step_gap_restarts_count():
    assert direct([[3,2,1],[1,3,2],[3,2,1],[1,3,2],[1,3,2]])==[10,10,10,10,11]
    m=model(ConfirmedAnchorPhase);d=dict(selection_margins=np.array([1,3,1,3]),tracks=np.array([10,11,10,11]),object_frames=np.array([0,0,2,2]),confidence=np.ones(4),anchor_confirmation_state={})
    assert m.select_candidate(d,np.array([0,1]),10,1)==0
    assert m.select_candidate(d,np.array([2,3]),10,1)==2


def test_target_missing_still_allows_anchor_confirmation_and_inputs_unchanged():
    d=fixture();d['roles'][6:8]=99;saved=copy.deepcopy(d);m=model(ConfirmedAnchorPhase);p,v,c,x=m.transform(d)
    assert not v[1] and d['tracks'][c[1,0]]==10 and d['tracks'][c[2,0]]==11
    for k in d:np.testing.assert_array_equal(d[k],saved[k])
    now=model(SemanticPriorityAnchorPhase).transform(d)
    np.testing.assert_array_equal(v,now[1]);np.testing.assert_array_equal(c[:,1],now[2][:,1])
    for a,b in zip(m.transform(d),(p,v,c,x)):np.testing.assert_array_equal(a,b)


def test_every_prefix_future_independence_and_one_confirmation_equivalence():
    d=fixture();m=model(ConfirmedAnchorPhase);full=m.transform(d)
    for end in range(1,6):
        prefix={k:v[:end] if k=='indices' else v[d['object_frames']<end] if k!='frame_count' else v for k,v in d.items()}
        for a,b in zip(m.transform(prefix),full):np.testing.assert_array_equal(a,b[:end])
    other=copy.deepcopy(d);other['crop_features'][12:]=[0,1]
    for a,b in zip(m.transform(other),full):np.testing.assert_array_equal(a[:3],b[:3])
    m.confirmation_samples=1
    for a,b in zip(m.transform(d),model(SemanticPriorityAnchorPhase).transform(d)):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('value',[0,-1,1.5,True])
def test_invalid_confirmation(value):
    with pytest.raises(ValueError):ConfirmedAnchorPhase(np.eye(2),confirmation_samples=value)
