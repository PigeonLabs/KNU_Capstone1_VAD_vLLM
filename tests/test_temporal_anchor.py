import numpy as np
import pytest
from ipad_vad.temporal_anchor import TemporalAnchorPhase
from test_verified_anchor import prepared


def data(steps,tracks,positive):
    return {'roles':np.ones(len(steps),int),'object_frames':np.array(steps),'tracks':np.array(tracks),'crop_features':np.array([[1.,0.] if x else [0.,1.] for x in positive])}


def test_three_observation_median_and_gap_track_reset():
    m=TemporalAnchorPhase(np.eye(2),anchor_role=1)
    d=data([0,1,2,3,5,6,6],[1,1,1,1,1,1,2],[False,True,True,False,False,True,True])
    np.testing.assert_array_equal(m.margins(d),[-1,0,1,1,-1,0,1])
    np.testing.assert_array_equal(m.raw_margins(d),[-1,1,1,-1,-1,1,1])
    # No persistent state leaks into another video/call.
    np.testing.assert_array_equal(m.margins(data([0],[1],[True])),[1])


def test_duplicate_same_sample_track_is_rejected_and_order_is_irrelevant():
    m=TemporalAnchorPhase(np.eye(2),anchor_role=1)
    with pytest.raises(ValueError):m.margins(data([0,0],[1,1],[True,False]))
    d=data([0,1,2],[1,1,1],[True,False,False]);expected=m.margins(d);order=[2,0,1]
    permuted={k:v[order] for k,v in d.items()};np.testing.assert_array_equal(m.margins(permuted),expected[order])
    with pytest.raises(ValueError):TemporalAnchorPhase(np.eye(2),window=0)


def test_prefix_and_future_invariance_and_other_roles_preserved():
    m=TemporalAnchorPhase(np.eye(2),anchor_role=1);d=data([0,1,2,3],[1,1,1,1],[True,False,True,False]);full=m.margins(d)
    for n in range(1,5):np.testing.assert_array_equal(m.margins({k:v[:n] for k,v in d.items()}),full[:n])
    d['crop_features'][3]=[1,0];np.testing.assert_array_equal(m.margins(d)[:3],full[:3])
    d['roles'][:]=0;np.testing.assert_array_equal(m.margins(d),m.raw_margins(d))


def test_relation_transform_is_causal_and_inputs_unchanged():
    d=prepared();m=TemporalAnchorPhase(np.eye(2));m.fit([d]);saved={k:v.copy() for k,v in d.items()};before=m.transform(d)
    changed={k:v.copy() for k,v in d.items()};changed['crop_features'][80:]=[0,1];after=m.transform(changed)
    for a,b in zip(before,after):np.testing.assert_array_equal(a[:40],b[:40])
    for k in d:np.testing.assert_array_equal(d[k],saved[k])
