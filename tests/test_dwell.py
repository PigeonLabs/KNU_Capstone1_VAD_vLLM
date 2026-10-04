import numpy as np
from ipad_vad.dwell import observed_ages,complete_runs,NormalDwell
from ipad_vad.dwell_scoring import DwellBaseline
from ipad_vad.process_calibration import StateCalibratedBaseline
from test_baseline import fixture
from test_process_calibration import model


def data(phases,valid=None):
    return {'phases':np.array(phases),'indices':np.arange(len(phases))*4,
            'relation_valid':np.ones(len(phases),bool) if valid is None else np.array(valid,bool)}


def test_only_complete_runs_with_observed_entry_exit_are_fit():
    d=data([0,0,1,1,2,2,0])
    assert list(complete_runs(d))==[(1,8.),(2,8.)]
    d['relation_valid'][1]=False
    assert list(complete_runs(d))==[(2,8.)], 'Invalid pre-entry observation excludes the run'
    d['relation_valid'][4]=False
    assert list(complete_runs(d))==[], 'Invalid exit or interior observation excludes the run'
    m=NormalDwell(minimum_complete_runs=2);m.fit([data([0,1,1,2,0]),data([0,1,1,0])])
    assert set(m.durations)=={1}, 'Insufficient support must not fall back to pooled durations'


def test_age_is_causal_and_gap_requires_new_observed_entry():
    d=data([0,0,1,1,1,1,1,2,2],[1,1,1,0,1,1,1,1,1]);d['indices'][-1]=40
    ages,known=observed_ages(d)
    assert known.tolist()==[False,False,True,False,False,False,False,True,True]
    assert ages[-1]==12, 'Use actual source-frame difference'
    changed={k:v.copy() for k,v in d.items()};changed['phases'][7:]=0;changed['relation_valid'][7:]=False
    a,k=observed_ages(changed);np.testing.assert_array_equal(ages[:7],a[:7]);np.testing.assert_array_equal(known[:7],k[:7])


def test_survival_is_long_tail_and_calibration_excludes_invalid_entries():
    m=NormalDwell(minimum_complete_runs=2,minimum_calibration_samples=2)
    m.fit([data([0,1,1,0]),data([0,1,1,1,0])]);cal=data([0,0,1,1,1,1,0]);m.calibrate([cal])
    assert len(m.reference)==4
    probe=data([0,1,1,1,1,1,1,0]);raw,valid,age,reason=m.raw(probe)
    assert not valid[0] and not valid[-1] and reason[-1]==3
    assert np.all(np.diff(raw[valid])>=0) and raw[-2]>raw[1]
    assert raw[-2]==np.log(3), 'Add-one survival remains finite beyond maximum duration'
    before=m.reference.copy();m.score(probe);np.testing.assert_array_equal(before,m.reference)


def test_dwell_fusion_preserves_visual_transition_and_abstains():
    base_template=model();cfg=dict(base_template.cfg,score_fusion='max',normal_dwell={'minimum_complete_runs':10,'minimum_calibration_samples':10})
    def f(seed):
        d=fixture(seed);d['relation_valid']=np.ones(len(d['phases']),bool);return d
    fit=[f(0),f(1)];cal=[f(2)];base=StateCalibratedBaseline(cfg,base_template.process);aug=DwellBaseline(cfg,base_template.process)
    for m in [base,aug]:m.fit(fit);m.calibrate(cal)
    probe=f(3);probe['phases'][5:15]=1;probe['relation_valid'][10]=False
    a=base.score(probe);b=aug.score(probe)
    np.testing.assert_array_equal(a['visual'],b['visual']);np.testing.assert_array_equal(a['process'],b['transition'])
    np.testing.assert_array_equal(a['combined'][~b['dwell_valid']],b['combined'][~b['dwell_valid']])
    assert np.all(b['combined']>=a['combined'])
    assert aug.threshold==np.quantile(aug.score(cal[0])['combined'],.99,method='higher')
