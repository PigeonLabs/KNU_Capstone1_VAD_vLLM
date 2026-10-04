import numpy as np
from ipad_vad.completed_dwell import CompletedContextDwell, CompletedDwellBaseline
from ipad_vad.context_dwell import ContextDwell
from ipad_vad.context_dwell_scoring import ContextDwellBaseline
from test_dwell import data
from test_baseline import fixture
from test_process_calibration import model


def test_complete_run_percentile_ties_and_calibration_independence():
    training=[data([0]+[2]*(length//4)+[0]) for length in [8,12,12,20]]
    m=CompletedContextDwell(minimum_complete_runs=2);m.fit(training);m.calibrate([])
    probe=data([0]+[2]*7+[0]);score,valid,age,_=m.score(probe)
    assert score[np.flatnonzero(age==12)[0]]==.5
    assert score[np.flatnonzero(age==20)[0]]==.875
    assert score[np.flatnonzero(age==24)[0]]==1
    assert np.all(score[valid&(age<=20)]<1)
    before=score.copy();m.calibrate([data([0]+[2]*1000+[0])]);np.testing.assert_array_equal(before,m.score(probe)[0])
    assert len(m.reference)==0
    changed={k:v.copy() for k,v in probe.items()};changed['phases'][5:]=3
    np.testing.assert_array_equal(before[:5],m.score(changed)[0][:5])
    old=ContextDwell(minimum_complete_runs=2);old.fit(training)
    np.testing.assert_array_equal(valid,old.raw(probe)[1])


def test_final_model_retains_branches_and_refits_q99():
    template=model();cfg=dict(template.cfg,score_fusion='max',normal_dwell={'minimum_complete_runs':10},dwell_context={'distribution':'entry'},dwell_score='fit_complete_percentile')
    def f(seed):
        d=fixture(seed);d['relation_valid']=np.ones(len(d['phases']),bool);return d
    original=ContextDwellBaseline(cfg,template.process);changed=CompletedDwellBaseline(cfg,template.process)
    for m in [original,changed]:m.fit([f(0),f(1)]);m.calibrate([f(2)])
    a=original.score(f(3));b=changed.score(f(3))
    for key in ['visual','transition','dwell_valid','dwell_age']:np.testing.assert_array_equal(a[key],b[key])
    assert changed.threshold==np.quantile(changed.score(f(2))['combined'],.99,method='higher')
    np.testing.assert_array_equal(b['combined'],np.maximum(b['visual'],np.where(b['dwell_valid'],np.maximum(b['transition'],b['dwell']),b['transition'])))
