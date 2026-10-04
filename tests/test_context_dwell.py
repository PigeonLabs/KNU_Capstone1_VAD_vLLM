import numpy as np
from ipad_vad.context_dwell import ContextDwell, observed_entry_context, complete_context_runs
from ipad_vad.dwell import NormalDwell, observed_ages, complete_runs
from ipad_vad.context_dwell_scoring import ContextDwellBaseline
from ipad_vad.dwell_scoring import DwellBaseline
from test_dwell import data
from test_baseline import fixture
from test_process_calibration import model


def test_entry_is_frozen_within_run_and_reset_by_gap():
    d=data([0,2,2,2,1,2,2,2],[1,1,1,1,1,0,1,1])
    context=observed_entry_context(d)
    assert context.tolist()==[-1,0,0,0,2,-1,-1,-1]
    age,known=observed_ages(d);np.testing.assert_array_equal(known,context>=0)
    assert [(key[1],duration) for key,duration in complete_context_runs(d)]==list(complete_runs(d))
    changed={k:v.copy() for k,v in d.items()};changed['phases'][4:]=0
    np.testing.assert_array_equal(context[:4],observed_entry_context(changed)[:4])


def test_context_duration_effect_is_separated_from_support_gate():
    training=[data([0,2,2,2,2,2,1,2,2,0])]*2
    entry=ContextDwell(minimum_complete_runs=2,minimum_calibration_samples=2)
    pooled=ContextDwell(distribution='pooled',minimum_complete_runs=2,minimum_calibration_samples=2)
    original=NormalDwell(minimum_complete_runs=2,minimum_calibration_samples=2)
    for m in [entry,pooled,original]:m.fit(training);m.calibrate(training)
    long=data([0,2,2,2,2,0]);short=data([1,2,2,2,2,0]);rare=data([3,2,2,2,2,0])
    for d in [long,short,rare]:
        np.testing.assert_array_equal(entry.raw(d)[1],pooled.raw(d)[1])
        a,valid,_,_=pooled.raw(d);np.testing.assert_array_equal(a[valid],original.raw(d)[0][valid])
    assert entry.raw(short)[0][4]>entry.raw(long)[0][4]
    assert pooled.raw(short)[0][4]==pooled.raw(long)[0][4]
    assert not entry.raw(rare)[1].any(), 'Unsupported entry has no pooled fallback in either arm'
    assert entry.raw(rare)[3][1]==3


def test_context_arms_keep_transition_and_visual_identical():
    template=model();cfg=dict(template.cfg,score_fusion='max',normal_dwell={'minimum_complete_runs':10,'minimum_calibration_samples':10})
    def f(seed):
        d=fixture(seed);d['relation_valid']=np.ones(len(d['phases']),bool);return d
    fit=[f(0),f(1)];cal=[f(2)];probe=f(3)
    original=DwellBaseline(cfg,template.process);original.fit(fit);original.calibrate(cal);before=original.score(probe);results=[]
    for mode in ['entry','pooled']:
        m=ContextDwellBaseline(dict(cfg,dwell_context={'distribution':mode}),template.process);m.fit(fit);m.calibrate(cal);out=m.score(probe);results.append(out)
        for key in ['visual','transition']:np.testing.assert_array_equal(out[key],before[key])
        assert m.threshold==np.quantile(m.score(cal[0])['combined'],.99,method='higher')
    np.testing.assert_array_equal(results[0]['dwell_valid'],results[1]['dwell_valid'])
