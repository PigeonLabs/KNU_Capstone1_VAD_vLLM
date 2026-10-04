import numpy as np
from ipad_vad.scoring import Baseline


def fixture(seed):
    rng=np.random.default_rng(seed);n=40
    return {'indices':np.arange(n)*4,'global_features':rng.normal(size=(n,16)),
            'phases':np.arange(n)%2,'roles':np.zeros(n,dtype=int),'object_frames':np.arange(n),
            'crop_features':rng.normal(size=(n,16))}


def test_fit_and_calibration_are_finite_for_unsupported_phase():
    cfg={'minimum_phase_samples':10,'transition_laplace_alpha':1,'pca_variance':.95,
         'pca_max_rank':4,'visual_process_weight':.5,'calibration_quantile':.99}
    process={'phases':[{'id':x} for x in ['a','b','c']],'normal_order':['a','b','c'],'cyclic':True}
    model=Baseline(cfg,process);model.fit([fixture(0),fixture(1)]);model.calibrate([fixture(2)])
    data=fixture(3);data['phases'][:]=2
    before=model.transition.copy();result=model.score(data)
    assert np.isfinite(result['combined']).all()
    assert np.array_equal(before,model.transition), 'Test observations must not update normal transitions'
    assert model.fallback_counts
    assert 0<=model.threshold<=1
