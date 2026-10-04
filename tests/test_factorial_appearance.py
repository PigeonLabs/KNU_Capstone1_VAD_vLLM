import numpy as np
import pytest
from test_observed_appearance import setup


def build(a,b):
    m,d=setup();m.cfg.pop('appearance_conditioning');m.cfg.update(appearance_fit_conditioning='observed_relation' if a else 'phase',appearance_inference_conditioning='observed_relation' if b else 'phase')
    m.fit([d]);m.calibrate([d]);return m,d


def test_fit_banks_depend_only_on_a_and_inference_routing_only_on_b():
    models={k:build(*k)[0] for k in [(0,0),(0,1),(1,0),(1,1)]};_,d=setup()
    for a in [0,1]:
        left,right=models[a,0],models[a,1]
        for key in left.spaces:
            assert left.spaces[key].n==right.spaces[key].n
            np.testing.assert_array_equal(left.spaces[key].basis,right.spaces[key].basis)
        for b in [0,1]:
            m=models[a,b];assert m.spaces[-1,0].n==(10 if a else 20)
            expected=np.where(d['relation_valid'],d['phases'],-1) if b else d['phases']
            np.testing.assert_array_equal(m.appearance_phases(d),expected)
    for m in models.values():np.testing.assert_array_equal(m.transition,models[0,0].transition);np.testing.assert_array_equal(m.spaces[-1,-1].basis,models[0,0].spaces[-1,-1].basis)


def test_legacy_joint_modes_exactly_reproduced():
    for value,mode in [(0,'phase'),(1,'observed_relation')]:
        m,d=build(value,value);old,_=setup(mode);old.fit([d]);old.calibrate([d])
        for k in ['visual','process','combined']:np.testing.assert_array_equal(old.score(d)[k],m.score(d)[k])


def test_support_fallback_survives_when_explicit_missing_fallback_off():
    m,d=setup();m.cfg.update(appearance_fit_conditioning='observed_relation',appearance_inference_conditioning='phase');d['relation_valid'][0]=False;m.fit([d]);m.calibrate([d]);assert (-1,0) not in m.spaces
    assert m.appearance_phases(d)[11]==0 and not d['relation_valid'][11]
    assert m.appearance_space_key(-1,0)==(-1,-1)
    assert m.appearance_routes(d,-1,np.array([11]))[0]==0
    assert m.spaces[-1,-1].n==40


def test_raw_routes_match_calibration_and_inputs_unchanged_for_each_cell():
    for a,b in [(0,0),(0,1),(1,0),(1,1)]:
        m,d=build(a,b);saved={k:v.copy() for k,v in d.items()}
        for role,frames,r in m.raw(d)[0]:np.testing.assert_array_equal(r,m.calibration[role])
        for k in saved:np.testing.assert_array_equal(d[k],saved[k])
        with pytest.raises(ValueError):m.appearance_phases(d,stage='other')
