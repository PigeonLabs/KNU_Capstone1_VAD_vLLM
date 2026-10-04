import numpy as np
from ipad_vad.process_calibration import StateCalibratedBaseline
from ipad_vad.scoring import Baseline, empirical_percentile
from test_baseline import fixture


def model(minimum=2):
    cfg={'minimum_phase_samples':10,'transition_laplace_alpha':1,'pca_variance':.95,'pca_max_rank':4,
         'visual_process_weight':.5,'calibration_quantile':.99,
         'process_calibration':{'mode':'previous_state','minimum_support':minimum}}
    process={'phases':[{'id':x} for x in ['a','b','c']],'normal_order':['a','b','c'],'cyclic':True}
    return StateCalibratedBaseline(cfg,process)


def test_conditioning_uses_previous_state_excludes_sequence_start_and_falls_back():
    m=model();c=[{'phases':np.array([0,0,1,1])},{'phases':np.array([1,0,2])}]
    raw=[([],np.array([0,1,1,8.])),([],np.array([0,8,2.]))]
    m.process_reference=np.concatenate([x[1] for x in raw]);m.fit_process_calibration(c,raw)
    assert m.state_process_references[0].tolist()==[1,1,2]
    assert m.state_process_references[1].tolist()==[8,8]
    assert len(m.state_process_references[2])==0
    d={'phases':np.array([0,1,2,0])};rawtest=np.array([0,1,8,2.])
    out=m.calibrate_process(d,rawtest)
    # Two ties at 1 among three prior-state-0 scores; constant prior-state-1 is neutral.
    np.testing.assert_allclose(out[1:3],[1/3,.5])
    global_score=empirical_percentile(m.process_reference,rawtest)
    np.testing.assert_array_equal(out[[0,3]],global_score[[0,3]])
    assert m.conditioning_mask(d).tolist()==[False,True,True,False]
    m.minimum_support=3
    assert m.conditioning_mask(d).tolist()==[False,True,False,False]
    # Future phase/raw changes cannot change earlier calibrated observations.
    d['phases'][2:]=1;rawtest[2:]=99
    np.testing.assert_array_equal(out[:2],m.calibrate_process(d,rawtest)[:2])


def test_full_model_preserves_visual_raw_transition_and_fits_new_combined_threshold():
    changed=model(10);base=Baseline(changed.cfg,changed.process)
    fit=[fixture(0),fixture(1)];cal=[fixture(2),fixture(3)];probe=fixture(4)
    for m in [base,changed]:m.fit(fit);m.calibrate(cal)
    np.testing.assert_array_equal(base.transition,changed.transition)
    np.testing.assert_array_equal(base.raw(probe)[1],changed.raw(probe)[1])
    a=base.score(probe);b=changed.score(probe)
    np.testing.assert_array_equal(a['visual'],b['visual'])
    np.testing.assert_array_equal(base.process_reference,changed.process_reference)
    assert a['process'][0]==b['process'][0]
    expected=np.concatenate([changed.score(d)['combined'] for d in cal])
    assert changed.threshold==np.quantile(expected,.99,method='higher')
    np.testing.assert_array_equal(b['combined'],.5*b['visual']+.5*b['process'])
    assert sum(map(len,changed.state_process_references.values()))==sum(len(d['phases'])-1 for d in cal)
