import numpy as np
import pytest
from ipad_vad.process_calibration import StateCalibratedBaseline
from ipad_vad.scoring import empirical_percentile


def template(population='consecutive_observed'):
    cfg={'process_calibration':{'minimum_support':2,'population':population},'transition_evidence_gate':'consecutive_observed'}
    p={'phases':[{'id':'a'},{'id':'b'},{'id':'c'}]}
    return StateCalibratedBaseline(cfg,p)


def test_observed_references_exclude_boundaries_and_missing_with_global_fallback():
    m=template();d={'phases':np.array([0,1,1,2,0,1]),'relation_valid':np.array([True,True,True,False,True,True])};raw=np.array([999.,2.,3.,100.,200.,4.])
    m.fit_process_calibration([d,d],[(None,raw),(None,raw)])
    np.testing.assert_array_equal(m.process_reference,[2,3,4,2,3,4]);np.testing.assert_array_equal(m.state_process_references[0],[2,4,2,4]);np.testing.assert_array_equal(m.state_process_references[1],[3,3]);assert len(m.state_process_references[2])==0
    probe={'phases':np.array([2,0])};s=m.calibrate_process(probe,np.array([0.,3.]));assert s[1]==empirical_percentile(m.process_reference,np.array([3.]))[0]
    # Changing excluded values cannot alter either kind of reference.
    changed=raw.copy();changed[[0,3,4]]=-999
    other=template();other.fit_process_calibration([d,d],[(None,changed),(None,changed)])
    np.testing.assert_array_equal(m.process_reference,other.process_reference)
    for k in m.state_process_references:np.testing.assert_array_equal(m.state_process_references[k],other.state_process_references[k])


def test_no_support_bad_population_gate_or_mask_are_explicit_failures():
    d={'phases':np.array([0,1]),'relation_valid':np.array([True,True])};r=[(None,np.array([0.,1.]))]
    with pytest.raises(ValueError,match='Insufficient'):template().fit_process_calibration([d],r)
    with pytest.raises(ValueError,match='Unknown'):template('bad').fit_process_calibration([d],r)
    m=template();m.cfg.pop('transition_evidence_gate')
    with pytest.raises(ValueError,match='matching'):m.fit_process_calibration([d],r)
    with pytest.raises(ValueError,match='Boolean'):template().fit_process_calibration([dict(d,relation_valid=np.ones(2))],r)
