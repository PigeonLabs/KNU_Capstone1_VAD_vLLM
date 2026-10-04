import numpy as np
import pytest
from ipad_vad.request_calibration import calibration_codes,request_residuals
from test_observed_appearance import setup


def make(mode,dispatch,d):
    m,_=setup();m.cfg.update(appearance_fit_conditioning='observed_relation',appearance_inference_conditioning=mode,appearance_request_calibration='dual_full_normal',appearance_calibration_dispatch=dispatch)
    m.fit([d]);m.calibrate([d]);return m


def test_actual_bank_unifies_support_fallback_without_changing_references():
    _,d=setup();d['sequence_id']='normal/a';d['relation_valid'][:20]=False
    hold=make('phase','actual_bank',d);pool=make('observed_relation','actual_bank',d);legacy=make('phase','request',d)
    assert (-1,0) not in hold.spaces
    for key,v in hold.request_calibration.references.items():
        np.testing.assert_array_equal(v,pool.request_calibration.references[key]);np.testing.assert_array_equal(v,legacy.request_calibration.references[key])
    for (r,f,a),(_,_,b) in zip(hold.score(d)['objects'],pool.score(d)['objects']):
        equal=np.array([hold.appearance_space_key(r,hold.appearance_phases(d)[i])==pool.appearance_space_key(r,pool.appearance_phases(d)[i]) for i in f])
        np.testing.assert_array_equal(a[equal],b[equal]);assert equal[:20].all() and not equal[30:].any()
    for role,frames,_,pooled in request_residuals(hold,d):
        actual=next(v for r,_,v in hold.raw(d)[0] if r==role);np.testing.assert_array_equal(actual[:20],pooled[:20]);np.testing.assert_array_equal(calibration_codes(hold,d,role,frames)[:20],np.zeros(20))
    for (r,f,a),(_,_,b) in zip(hold.score(d)['objects'],legacy.score(d)['objects']):np.testing.assert_array_equal(a[20:],b[20:])


def test_role_specific_support_is_used_and_global_fallback_is_pooled():
    _,d=setup();d['sequence_id']='normal/a';m=make('phase','actual_bank',d)
    # Simulate a role whose phase bank has insufficient support while global is supported.
    del m.spaces[0,0]
    frames=np.arange(len(d['indices']));assert calibration_codes(m,d,-1,frames)[0]==1;assert calibration_codes(m,d,0,frames)[0]==0
    assert np.all(calibration_codes(m,d,99,frames)==0)


def test_invalid_or_orphan_dispatch_rejected():
    _,d=setup();d['sequence_id']='normal/a'
    with pytest.raises(ValueError):make('phase','bad',d)
    m,_=setup();m.cfg['appearance_calibration_dispatch']='actual_bank';m.fit([d])
    with pytest.raises(ValueError):m.calibrate([d])
