import numpy as np
import pytest
from test_observed_appearance import setup


def make(seed=0):
    m,d=setup('phase');m.cfg['appearance_fit_sampling']={'mode':'count_matched_random','seed':seed}
    return m,d


def test_counts_unique_reproducibility_and_pooled_preservation():
    m,d=make();same,_=make();other,_=make(1);full,_=setup('phase');observed,_=setup()
    for model in [m,same,other,full,observed]:model.fit([d])
    assert set(m.spaces)==set(observed.spaces)
    for key,idx in m.sampling_indices.items():
        assert len(idx)==10 and len(np.unique(idx))==10 and np.all(np.diff(idx)>0)
        assert idx.min()>=0 and idx.max()<20
        np.testing.assert_array_equal(idx,same.sampling_indices[key])
        assert m.spaces[key].n==observed.spaces[key].n
    assert any(not np.array_equal(v,other.sampling_indices[k]) for k,v in m.sampling_indices.items())
    for role in [-1,0]:
        for name in ['mean','basis']:np.testing.assert_array_equal(getattr(m.spaces[role,-1],name),getattr(full.spaces[role,-1],name))
    np.testing.assert_array_equal(m.transition,full.transition)
    np.testing.assert_array_equal(m.raw(d)[1],full.raw(d)[1])


def test_only_counts_not_mask_identity_affect_sampling():
    m,d=make();other,_=make();changed={k:v.copy() for k,v in d.items()};changed['relation_valid']=~changed['relation_valid']
    m.fit([d]);other.fit([changed])
    for key in m.spaces:
        for name in ['mean','basis']:np.testing.assert_array_equal(getattr(m.spaces[key],name),getattr(other.spaces[key],name))


def test_zero_and_low_support_keep_full_pooled_and_clear_refit_state():
    m,d=make();m.fit([d]);d['relation_valid'][:20]=False;d['relation_valid'][20:27]=False;m.fit([d])
    assert set(m.spaces)=={(-1,-1),(0,-1)}
    for role in [-1,0]:
        assert len(m.sampling_indices[role,0])==0 and len(m.sampling_indices[role,1])==3
        assert m.spaces[role,-1].n==40 and m.appearance_space_key(role,1)==(role,-1)


@pytest.mark.parametrize('spec',[{'mode':'bad','seed':0},{'mode':'count_matched_random','seed':True},{'mode':'count_matched_random','seed':-1},{'mode':'count_matched_random','seed':.5}])
def test_invalid_sampling_rejected(spec):
    m,d=make();m.cfg['appearance_fit_sampling']=spec
    with pytest.raises(ValueError):m.fit([d])


def test_reject_restricted_population_and_invalid_mask():
    m,d=make();m.cfg['appearance_fit_conditioning']='observed_relation'
    with pytest.raises(ValueError):m.fit([d])
    del m.cfg['appearance_fit_conditioning'];d['relation_valid']=d['relation_valid'].astype(int)
    with pytest.raises(ValueError):m.fit([d])
