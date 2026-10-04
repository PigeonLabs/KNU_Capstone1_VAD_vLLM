import numpy as np
from ipad_vad.scoring import Subspace, empirical_percentile
from ipad_vad.tracking import Tracker


def test_orthogonal_defect_exceeds_normal_residual():
    x=np.column_stack([np.arange(20),np.zeros(20),np.zeros(20)])
    model=Subspace(x,max_rank=1)
    assert model.residual([[5,0,0]])[0]<1e-8
    assert model.residual([[5,0,3]])[0]>8.9


def test_constant_calibration_is_neutral():
    assert empirical_percentile([0,0,0],[0,1]).tolist()==[.5,1.]


def test_tracker_matches_roles_and_expires():
    t=Tracker(.2,2)
    assert t.update([[0,0,10,10]],[0],0).tolist()==[0]
    assert t.update([[1,0,11,10]],[0],1).tolist()==[0]
    assert t.update([[1,0,11,10]],[1],2).tolist()==[1]
    assert t.update([[1,0,11,10]],[0],5).tolist()==[2]
