import numpy as np
import pytest
from ipad_vad.motion_anchor import select_role_anchors,fit_motion_axis
from test_spatial_phase import fixture


def test_anchor_selection_does_not_require_r01_corridor_and_preserves_id():
    d=fixture();d['boxes'][:,[1,3]]+=.4
    chosen=select_role_anchors(d,0)
    np.testing.assert_array_equal(chosen,np.arange(30))
    changed={k:v.copy() for k,v in d.items()};changed['confidence'][20:]=0
    np.testing.assert_array_equal(select_role_anchors(changed,0)[:20],chosen[:20])
    assert np.all(select_role_anchors(d,1)==-1)


def test_axis_is_fitted_to_supported_normal_motion_and_static_data_fails():
    d=fixture();d['boxes']=d['boxes'][:,[1,0,3,2]]
    axis,evidence=fit_motion_axis([d],0)
    assert axis==1 and evidence['moving_tracks']==1
    d['boxes'][:]=[.2,.2,.3,.3]
    with pytest.raises(ValueError):fit_motion_axis([d],0)
