import numpy as np
from ipad_vad.spatial_phase import SpatialPhase


def fixture():
    x=np.linspace(.05,.95,30);boxes=np.array([[v-.02,.38,v+.02,.42] for v in x])
    return {'indices':np.arange(30)*4,'object_frames':np.arange(30),'roles':np.zeros(30,dtype=int),
            'tracks':np.zeros(30,dtype=int),'boxes':boxes,'confidence':np.ones(30)}


def test_future_boxes_do_not_change_earlier_phases():
    model=SpatialPhase();normal=fixture();model.fit([normal]);test=fixture()
    a=model.transform(test)[0]
    test['boxes'][20:]=[.02,.38,.06,.42]
    b=model.transform(test)[0]
    assert np.array_equal(a[:20],b[:20])
    assert len(np.unique(a))==3


def test_off_path_static_object_is_not_phase_anchor():
    model=SpatialPhase();normal=fixture();model.fit([normal]);test=fixture()
    test['boxes'][:]=[.8,.8,.9,.9]
    phases,observed,_,_=model.transform(test)
    assert not observed.any()
    assert np.all(phases==0)
