import numpy as np
from ipad_vad.product_roi import fit_product_roi,pixel_roi,restore_boxes
from test_spatial_phase import fixture


def test_roi_ignores_static_distractor_and_preserves_travel_axis():
    normal=fixture()
    for k,v in {'boxes':np.tile([.8,.8,.9,.9],(30,1)), 'roles':np.zeros(30,dtype=int),'tracks':np.ones(30,dtype=int)}.items():
        normal[k]=np.concatenate([normal[k],v])
    roi,evidence=fit_product_roi([normal])
    np.testing.assert_allclose(roi,[0,.35,1,.45])
    assert evidence['dynamic_tracks']==1


def test_crop_coordinate_mapping_including_empty_boxes():
    rect=pixel_roi([0,.351,1,.459],320,240)
    assert rect.tolist()==[0,84,320,111]
    np.testing.assert_allclose(restore_boxes([[10,2,30,20]],rect,320,240),[[10,86,30,104]])
    assert restore_boxes(np.empty((0,4)),rect,320,240).shape==(0,4)
