import copy
import numpy as np
import pytest
from ipad_vad.dwell_scoring import observed_transition_mask
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from test_process_calibration import model
from test_baseline import fixture


def test_transition_boundaries_prefix_and_invalid_mask():
    d={'phases':np.array([0,0,1,1,2,2,2]),'relation_valid':np.array([True,True,False,True,True,False,True])}
    expected=np.array([False,True,False,False,True,False,False]);np.testing.assert_array_equal(observed_transition_mask(d),expected)
    for n in range(8):np.testing.assert_array_equal(observed_transition_mask({k:v[:n] for k,v in d.items()}),expected[:n])
    for invalid in [np.ones(7),np.ones(6,bool),np.ones((7,1),bool)]:
        with pytest.raises(ValueError):observed_transition_mask(dict(d,relation_valid=invalid))


def test_gate_preserves_references_visual_and_dwell_but_removes_unobserved_transition():
    template=model();cfg=dict(template.cfg,score_fusion='max',normal_dwell={'minimum_complete_runs':10},dwell_context={'distribution':'entry'},dwell_score='fit_lognormal_cdf',dwell_lognormal={'sigma_floor':.05})
    def f(seed):
        d=fixture(seed);d['relation_valid']=np.ones(len(d['phases']),bool);return d
    old=LognormalDwellBaseline(cfg,template.process);old.fit([f(0),f(1)]);new=copy.deepcopy(old);new.cfg=dict(cfg,transition_evidence_gate='consecutive_observed')
    cal=f(2);cal['relation_valid'][4:8]=False
    for m in [old,new]:m.calibrate([cal])
    for key in old.calibration:np.testing.assert_array_equal(old.calibration[key],new.calibration[key])
    for key in old.state_process_references:np.testing.assert_array_equal(old.state_process_references[key],new.state_process_references[key])
    probe=f(3);probe['relation_valid'][2:5]=False;a=old.score(probe);b=new.score(probe)
    for key in ['visual','transition','dwell','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']:np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_array_equal(b['transition_raw'],old.raw(probe)[1]);assert np.all(b['transition_gated'][[0,2,3,4,5]]==0);assert np.all(b['combined']<=a['combined'])
    assert new.threshold==np.quantile(new.score(cal)['combined'],.99,method='higher')
    new.cfg['transition_evidence_gate']='bad'
    with pytest.raises(ValueError):new.score(probe)
