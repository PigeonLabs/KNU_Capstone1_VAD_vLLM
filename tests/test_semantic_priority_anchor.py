import copy
import numpy as np
from ipad_vad.semantic_priority_anchor import SemanticPriorityAnchorPhase
from ipad_vad.temporal_anchor import TemporalAnchorPhase


def fixture():
    n=5
    return dict(indices=np.arange(n)*4,frame_count=np.array(n*4),
        roles=np.tile([1,1,0,0],n),object_frames=np.repeat(np.arange(n),4),
        tracks=np.tile([10,11,20,21],n),confidence=np.tile([.9,.3,.8,.4],n),
        boxes=np.tile([[.1,.1,.3,.3],[.5,.1,.7,.3],[.3,.4,.4,.5],[.6,.4,.7,.5]],(n,1)),
        crop_features=np.array([[1.,0.],[.8,.6],[1.,0.],[1.,0.]]+[[.8,.6],[1.,0.],[1.,0.],[1.,0.]]*(n-1)))


def model(cls):
    m=cls(np.eye(2),anchor_role=1,target_role=0,window=1,reset_on_track_change=True)
    m.area_upper={1:.1,0:.1};m.location=np.zeros(6);m.scale=np.ones(6);m.centers=np.zeros((4,6));return m


def test_semantic_priority_ties_empty_and_target_policy():
    m=model(SemanticPriorityAnchorPhase);d=fixture();d['selection_margins']=np.ones(20)
    assert m.select_candidate(d,np.array([],int),10,1)==-1
    assert m.select_candidate(d,np.array([0,1]),11,1)==1
    d['selection_margins'][0]=2
    assert m.select_candidate(d,np.array([0,1]),11,1)==0
    d['selection_margins'][0]=1;d['confidence'][:2]=.5
    assert m.select_candidate(d,np.array([1,0]),None,1)==0
    d['selection_margins'][3]=10
    assert m.select_candidate(d,np.array([2,3]),20,0)==2


def test_priority_changes_anchor_not_eligibility_or_target_and_resets_history():
    d=fixture();saved=copy.deepcopy(d);old=model(TemporalAnchorPhase);new=model(SemanticPriorityAnchorPhase)
    a=old.transform(d);b=new.transform(d)
    np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2][:,1],b[2][:,1])
    assert a[2][1,0]==4 and b[2][1,0]==5
    raw=copy.deepcopy(new);raw.smoothing=1
    np.testing.assert_array_equal(b[3][1],raw.transform(d)[3][1])
    for k in d:np.testing.assert_array_equal(d[k],saved[k])
    # Rejection remains identical even when strongest semantic candidate is oversized.
    d['boxes'][5]=[0,0,1,1];d['crop_features'][8:10]=[0,1]
    a=old.transform(d);b=new.transform(d)
    assert b[2][1,0]==4 and not b[1][2]
    np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2][:,1],b[2][:,1])


def test_every_prefix_and_future_changes_leave_past_invariant():
    d=fixture();m=model(SemanticPriorityAnchorPhase);m.window=3;full=m.transform(d)
    for end in range(1,6):
        prefix={k:v[:end] if k=='indices' else v[d['object_frames']<end] if k not in ['frame_count'] else v for k,v in d.items()}
        for a,b in zip(m.transform(prefix),full):np.testing.assert_array_equal(a,b[:end])
    d['crop_features'][12:]=[0,1]
    for a,b in zip(m.transform(d),full):np.testing.assert_array_equal(a[:3],b[:3])
