import numpy as np
import pytest
from ipad_vad.scoring import Baseline, empirical_percentile


def setup(mode='observed_relation'):
    cfg={'appearance_conditioning':mode,'minimum_phase_samples':10,'pca_variance':.95,'pca_max_rank':2,'transition_laplace_alpha':1.,'calibration_quantile':.99,'score_fusion':'max'}
    process={'phases':[{'id':'a'},{'id':'b'}],'normal_order':['a','b'],'cyclic':True}
    rng=np.random.default_rng(42);n=40
    d={'indices':np.arange(n)*4,'phases':np.repeat([0,1],20),'relation_valid':np.tile(np.arange(20)<10,2),'global_features':rng.normal(size=(n,4)), 'roles':np.zeros(n,int),'object_frames':np.arange(n),'crop_features':rng.normal(size=(n,4))}
    return Baseline(cfg,process),d


def test_observed_fit_exclusion_pooled_once_and_process_unchanged():
    model,d=setup();old,_=setup('phase');original=d['phases'].copy();model.fit([d]);old.fit([d])
    for role in [-1,0]:
        assert model.spaces[role,0].n==model.spaces[role,1].n==10
        assert model.spaces[role,-1].n==40
        np.testing.assert_array_equal(model.spaces[role,-1].mean,old.spaces[role,-1].mean)
        np.testing.assert_array_equal(model.spaces[role,-1].basis,old.spaces[role,-1].basis)
    np.testing.assert_array_equal(model.transition,old.transition)
    np.testing.assert_array_equal(model.raw(d)[1],old.raw(d)[1])
    np.testing.assert_array_equal(d['phases'],original)
    # Changing ONLY unobserved features cannot alter any observed phase bank.
    changed={k:v.copy() for k,v in d.items()};changed['global_features'][~d['relation_valid']]+=100
    other,_=setup();other.fit([changed])
    for p in [0,1]:np.testing.assert_array_equal(other.spaces[-1,p].mean,model.spaces[-1,p].mean)


def test_minimum_support_and_global_pool_fallback():
    model,d=setup();d['relation_valid'][0]=False;model.fit([d])
    assert (-1,0) not in model.spaces and (0,0) not in model.spaces
    assert model.appearance_space_key(0,0)==(0,-1)
    assert model.appearance_space_key(99,0)==(-1,-1)
    assert model.appearance_space_key(99,-1)==(-1,-1)
    assert model.appearance_space_key(0,1)==(0,1)
    # Unseen role with the same feature dimension uses global pooled residuals.
    test={k:v.copy() for k,v in d.items()};test['roles'][:]=99
    for role,frames,r in model.raw(test)[0]:
        if role==99:np.testing.assert_allclose(r,model.spaces[-1,-1].residual(test['crop_features']))


def test_calibration_and_inference_share_observation_route():
    model,d=setup();model.fit([d]);model.calibrate([d]);out=model.score(d)
    for role,frames,r in model.raw(d)[0]:
        x=d['global_features'] if role==-1 else d['crop_features'];expected=np.empty(len(x))
        for i in range(len(x)):
            key=(role,int(d['phases'][i])) if d['relation_valid'][i] else (role,-1)
            expected[i]=model.spaces[key].residual(x[i:i+1])[0]
        np.testing.assert_allclose(r,expected,atol=1e-12)
        np.testing.assert_array_equal(model.calibration[role],r)
        found=[s for rid,_,s in out['objects'] if rid==role][0]
        np.testing.assert_array_equal(found,empirical_percentile(r,r))
    assert model.threshold==np.quantile(out['combined'],.99,method='higher')


def test_all_observed_matches_legacy_and_invalid_mask_rejected():
    model,d=setup();d['relation_valid'][:]=True;old,_=setup('phase')
    for m in [old,model]:m.fit([d]);m.calibrate([d])
    for k in ['visual','process','combined']:np.testing.assert_array_equal(model.score(d)[k],old.score(d)[k])
    d['relation_valid']=d['relation_valid'].astype(int)
    with pytest.raises(ValueError):model.appearance_phases(d)
    model.cfg['appearance_conditioning']='unknown'
    with pytest.raises(ValueError):model.appearance_phases(d)
