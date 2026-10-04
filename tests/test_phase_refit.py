import copy
import numpy as np
import pytest
from ipad_vad.phase_refit import refit_phase_centers
from ipad_vad.relational_phase import RelationalPhase
from test_relational_phase import fixture


def test_existing_fit_recipe_exact_and_no_gate_reestimation(monkeypatch):
    d=fixture();old=RelationalPhase();old.fit([d]);new=RelationalPhase();new.area_upper=copy.deepcopy(old.area_upper)
    def forbidden(_):raise AssertionError('Gate must remain frozen')
    monkeypatch.setattr(new,'fit_gates',forbidden)
    frozen=copy.deepcopy(d);refit_phase_centers(new,[d])
    for key in ['location','scale','centers']:np.testing.assert_array_equal(getattr(new,key),getattr(old,key))
    for a,b in zip(new.transform(d),old.transform(d)):np.testing.assert_array_equal(a,b)
    for k in d:np.testing.assert_array_equal(d[k],frozen[k])
    assert new.area_upper==old.area_upper


def test_supplied_gate_and_selection_survive_refitting():
    d=fixture();m=RelationalPhase();m.fit([d]);m.area_upper={0:.1,1:.009};before=m.descriptors(d)
    refit_phase_centers(m,[d]);assert m.area_upper=={0:.1,1:.009}
    for a,b in zip(m.descriptors(d),before):np.testing.assert_array_equal(a,b)


def test_fitted_phase_ignores_future_and_inference_position():
    d=fixture();m=RelationalPhase();m.fit_gates([d]);refit_phase_centers(m,[d]);full=m.transform(d)
    for end in [1,21,40,79]:
        prefix={k:v[:end] if k=='indices' else v[d['object_frames']<end] if k!='frame_count' else v for k,v in d.items()}
        for a,b in zip(m.transform(prefix),full):np.testing.assert_array_equal(a,b[:end])
    other=copy.deepcopy(d);other['indices']=other['indices']*5+1000;other['frame_count']=np.array(99999)
    for a,b in zip(m.transform(other),full):np.testing.assert_array_equal(a,b)
    other=copy.deepcopy(d);other['boxes'][80:]=[0,0,1,1]
    for a,b in zip(m.transform(other),full):np.testing.assert_array_equal(a[:40],b[:40])


def test_failed_refit_does_not_replace_saved_model():
    d=fixture();m=RelationalPhase();m.fit([d]);saved=copy.deepcopy(m);bad=copy.deepcopy(d);bad['roles'][:]=99
    with pytest.raises(ValueError,match='Insufficient'):refit_phase_centers(m,[bad])
    for k in ['location','scale','centers']:np.testing.assert_array_equal(getattr(m,k),getattr(saved,k))


def test_missing_or_empty_fit_is_rejected():
    with pytest.raises(ValueError,match='caches'):refit_phase_centers(RelationalPhase(),[])
    with pytest.raises(ValueError,match='gates'):refit_phase_centers(RelationalPhase(),[fixture()])
