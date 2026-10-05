import copy
import numpy as np
import pytest
from ipad_vad.dwell_evidence import same_pair_since_entry_mask
from ipad_vad.transition_evidence import same_track_pair_transition_mask
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from test_pair_transition_gate import prefix
from test_baseline import fixture
from test_process_calibration import model


def data(phases, pairs, valid=None):
    n=len(phases)
    return dict(phases=np.array(phases), tracks=np.array(pairs).reshape(-1),
                object_frames=np.repeat(np.arange(n),2),
                relation_detection_indices=np.arange(2*n).reshape(n,2),
                relation_valid=np.ones(n,bool) if valid is None else np.array(valid,bool))


def test_entry_continuity_track_changes_and_new_entry():
    d=data([0,0,1,1,1,1,2,2,3,3],[(1,2)]*4+[(1,3)]*3+[(4,3)]*3)
    expected=[False,False,True,True,False,False,True,False,True,True]
    np.testing.assert_array_equal(same_pair_since_entry_mask(d),expected)
    assert same_track_pair_transition_mask(d)[5] and not expected[5]
    # A changed pair at the phase boundary does not establish an entry.
    d['phases'][4:6]=2
    np.testing.assert_array_equal(same_pair_since_entry_mask(d)[4:8],[False]*4)


def test_missing_reacquisition_first_sample_and_empty():
    d=data([0,1,1,1,1,2],[(1,2)]*6,[True,True,False,True,True,True])
    np.testing.assert_array_equal(same_pair_since_entry_mask(d),[False,True,False,False,False,True])
    assert not same_pair_since_entry_mask(data([1,1],[(1,2)]*2)).any()
    assert len(same_pair_since_entry_mask(prefix(d,0)))==0
    d['relation_valid'][:]=False
    assert not same_pair_since_entry_mask(d).any()


def test_prefix_future_independence_and_video_reset():
    d=data([0,1,1,1,2,2],[(1,2)]*6)
    before=copy.deepcopy(d);full=same_pair_since_entry_mask(d)
    for n in range(7):np.testing.assert_array_equal(same_pair_since_entry_mask(prefix(d,n)),full[:n])
    changed=copy.deepcopy(d);changed['tracks'][8:]=9;changed['phases'][4:]=0
    np.testing.assert_array_equal(same_pair_since_entry_mask(changed)[:4],full[:4])
    assert not same_pair_since_entry_mask(data([1,1],[(1,2)]*2)).any()
    for key in d:np.testing.assert_array_equal(d[key],before[key])


@pytest.mark.parametrize('mutation',['valid_type','selected_missing','misaligned','negative_track'])
def test_invalid_metadata_rejected(mutation):
    d=data([0,1],[(1,2)]*2)
    if mutation=='valid_type':d['relation_valid']=np.ones(2,int)
    elif mutation=='selected_missing':d['relation_detection_indices'][0,0]=-1
    elif mutation=='misaligned':d['relation_detection_indices'][1,0]=0
    else:d['tracks'][0]=-1
    with pytest.raises(ValueError):same_pair_since_entry_mask(d)


def test_fusion_keeps_fit_raw_scores_references_and_recalibrates_q99():
    t=model();cfg=dict(t.cfg,score_fusion='max',normal_dwell={'minimum_complete_runs':2},
        dwell_context={'distribution':'entry'},dwell_score='fit_lognormal_cdf',
        dwell_lognormal={'sigma_floor':.05},transition_evidence_gate='same_track_pair')
    def f(seed):
        d=fixture(seed);n=len(d['phases']);d['phases']=np.tile([0,1,1,0,0,1,1,1],5)
        d.update(relation_valid=np.ones(n,bool),tracks=np.tile([10,20],n),relation_detection_indices=np.arange(n*2).reshape(n,2))
        d['object_frames']=np.repeat(np.arange(n),2);d['roles']=np.zeros(n*2,int);d['crop_features']=np.repeat(d['crop_features'],2,axis=0)
        return d
    old=LognormalDwellBaseline(cfg,t.process);old.fit([f(0),f(1)])
    control=copy.deepcopy(old);control.cfg=dict(cfg,dwell_evidence_gate='ungated')
    new=copy.deepcopy(old);new.cfg=dict(cfg,dwell_evidence_gate='same_track_pair_since_entry')
    cal=f(2);cal['tracks'][4::2]=30
    for m in [old,control,new]:m.calibrate([cal])
    a,b,c=old.score(cal),control.score(cal),new.score(cal)
    for key in a:
        if key=='objects':continue
        np.testing.assert_array_equal(a[key],b[key])
        if key not in ['process','combined']:np.testing.assert_array_equal(a[key],c[key])
    for ref in ['calibration','state_process_references']:
        for key in getattr(old,ref):np.testing.assert_array_equal(getattr(old,ref)[key],getattr(new,ref)[key])
    assert old.dwell.log_parameters==new.dwell.log_parameters
    for key in old.dwell.context_durations:np.testing.assert_array_equal(old.dwell.context_durations[key],new.dwell.context_durations[key])
    mask=same_pair_since_entry_mask(cal)
    np.testing.assert_array_equal(c['dwell_gated'],np.where(mask&a['dwell_valid'],a['dwell'],0))
    assert np.any(a['dwell'][a['dwell_valid']&~mask]>0)
    np.testing.assert_array_equal(c['process'],np.maximum(a['transition_gated'],c['dwell_gated']))
    assert np.all(c['combined']<=a['combined'])
    np.testing.assert_array_equal(c['combined'][mask],a['combined'][mask])
    assert new.threshold==np.quantile(c['combined'],.99,method='higher')
    new.cfg['dwell_evidence_gate']='bad'
    with pytest.raises(ValueError,match='dwell evidence'):new.score(cal)
