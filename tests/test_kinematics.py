import numpy as np
from ipad_vad.kinematics import progress_signal,NormalProgress
from ipad_vad.kinematic_scoring import KinematicBaseline
from ipad_vad.scoring import Baseline
from test_baseline import fixture


def data():
    return {'indices':np.arange(6)*4,'boxes':np.array([[i*.04,.2,i*.04+.1,.4] for i in range(6)]),
            'tracks':np.array([0,0,0,1,1,1])}


def test_velocity_uses_source_frames_and_abstains_on_id_change_and_gap():
    d=data();chosen=np.array([0,1,-1,3,4,5]);v,valid,reason=progress_signal(d,chosen,0)
    np.testing.assert_allclose(v[valid],.01)
    assert valid.tolist()==[False,True,False,False,True,True]
    _,valid,reason=progress_signal(d,np.arange(6),0)
    assert not valid[3] and reason[3]==4
    before=progress_signal(d,np.arange(6),0)[0]
    d['boxes'][4:]=[.9,.2,1,.4]
    np.testing.assert_array_equal(before[:4],progress_signal(d,np.arange(6),0)[0][:4])


def sample(velocity,valid=None):
    v=np.asarray(velocity)
    return {'motion_velocity':v,'motion_valid':np.ones(len(v),bool) if valid is None else np.asarray(valid,bool),'phases':np.zeros(len(v),int)}


def test_invalid_sentinels_do_not_contaminate_fit_or_calibration_and_stop_scores_high():
    normal=sample(np.tile([.009,.01,.011],10));contaminated=sample(np.r_[normal['motion_velocity'],1000],np.r_[normal['motion_valid'],False])
    m=NormalProgress();m.fit([contaminated]);m.calibrate([contaminated])
    assert m.models[0]['samples']==30 and len(m.reference)==30
    score,valid=m.score(sample([.01,0,1000],[True,True,False]))
    assert score[1]>score[0] and not valid[2]


def test_augmented_fusion_preserves_visual_and_abstains_exactly():
    cfg={'minimum_phase_samples':10,'transition_laplace_alpha':1,'pca_variance':.95,'pca_max_rank':4,
         'visual_process_weight':.5,'calibration_quantile':.99,'normal_progress':{'minimum_samples':10,'scale_floor':1e-6}}
    process={'phases':[{'id':x} for x in ['a','b','c']],'normal_order':['a','b','c'],'cyclic':True}
    def f(seed):
        d=fixture(seed);d.update(motion_velocity=np.linspace(.009,.011,40),motion_valid=np.ones(40,bool));return d
    fit=[f(0),f(1)];cal=[f(2)]
    base=Baseline(cfg,process);base.fit(fit);base.calibrate(cal)
    augmented=KinematicBaseline(cfg,process);augmented.fit(fit);augmented.calibrate(cal)
    test=f(3);test['motion_valid'][::2]=False;test['motion_velocity'][1::2]=0
    a=base.score(test);b=augmented.score(test)
    np.testing.assert_array_equal(a['visual'],b['visual'])
    np.testing.assert_array_equal(a['process'],b['transition'])
    np.testing.assert_array_equal(a['combined'][::2],b['combined'][::2])
    assert np.all(b['process'][1::2]>=a['process'][1::2])
    q=np.quantile(augmented.score(cal[0])['combined'],.99,method='higher')
    assert augmented.threshold==q


def test_longer_progress_requires_entire_track_window_and_is_causal():
    d=data();d['tracks'][:]=0;d['indices']=np.array([0,4,12,16,20,24])
    v,valid,_=progress_signal(d,np.arange(6),0,lag_samples=3)
    assert valid.tolist()==[False,False,False,True,True,True]
    np.testing.assert_allclose(v[3],.12/16)
    earlier=v.copy();d['boxes'][5]=[.9,.2,1,.4]
    np.testing.assert_array_equal(earlier[:5],progress_signal(d,np.arange(6),0,lag_samples=3)[0][:5])
    chosen=np.arange(6);chosen[2]=-1
    assert not progress_signal(d,chosen,0,lag_samples=3)[1].any()
    d['tracks'][2]=1
    assert not progress_signal(d,np.arange(6),0,lag_samples=3)[1].any()
