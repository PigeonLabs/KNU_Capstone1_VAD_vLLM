import copy
import numpy as np
import pytest
from ipad_vad.transition_evidence import observed_transition_mask,same_track_pair_transition_mask
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from test_baseline import fixture
from test_process_calibration import model


def data():
    pairs=np.array([[10,20],[10,20],[11,20],[11,21],[12,22],[-1,-1],[12,22],[12,22],[-1,22],[12,22]])
    n=len(pairs);selected=np.arange(n*2).reshape(n,2);selected[pairs<0]=-1
    return {'phases':np.arange(n)%3,'relation_valid':np.all(pairs>=0,axis=1),'relation_detection_indices':selected,'tracks':np.maximum(pairs.ravel(),0),'object_frames':np.repeat(np.arange(n),2)}


def prefix(d,n):
    return {k:(v[d['object_frames']<n] if k in ['tracks','object_frames'] else v[:n]) for k,v in d.items()}


def test_pair_boundaries_missing_reacquisition_and_prefix():
    d=data();expected=np.array([False,True,False,False,False,False,False,True,False,False]);np.testing.assert_array_equal(same_track_pair_transition_mask(d),expected)
    assert np.all(~expected|observed_transition_mask(d))
    for n in range(len(expected)+1):np.testing.assert_array_equal(same_track_pair_transition_mask(prefix(d,n)),expected[:n])
    changed=copy.deepcopy(d);changed['tracks'][16:]=999;np.testing.assert_array_equal(same_track_pair_transition_mask(changed)[:8],expected[:8])
    # All missing, including an empty detection cache, must not gather sentinel -1.
    empty=dict(phases=np.zeros(3,int),relation_valid=np.zeros(3,bool),relation_detection_indices=np.full((3,2),-1,int),tracks=np.array([],int),object_frames=np.array([],int))
    assert not same_track_pair_transition_mask(empty).any()


@pytest.mark.parametrize('mutation',['float_selection','shape','negative','out_of_range','stale_frame','observed_missing','negative_track','float_track','frame_shape','future_frame'])
def test_invalid_selection_metadata_fails_explicitly(mutation):
    d=data()
    if mutation=='float_selection':d['relation_detection_indices']=d['relation_detection_indices'].astype(float)
    elif mutation=='shape':d['relation_detection_indices']=d['relation_detection_indices'][:,0]
    elif mutation=='negative':d['relation_detection_indices'][0,0]=-2
    elif mutation=='out_of_range':d['relation_detection_indices'][0,0]=len(d['tracks'])
    elif mutation=='stale_frame':d['relation_detection_indices'][1,0]=0
    elif mutation=='observed_missing':d['relation_detection_indices'][1,0]=-1
    elif mutation=='negative_track':d['tracks'][0]=-1
    elif mutation=='float_track':d['tracks']=d['tracks'].astype(float)
    elif mutation=='frame_shape':d['object_frames']=d['object_frames'][:-1]
    elif mutation=='future_frame':d['object_frames'][-1]=len(d['phases'])
    with pytest.raises(ValueError):same_track_pair_transition_mask(d)


def test_fusion_preserves_appearance_cdf_and_dwell():
    t=model();cfg=dict(t.cfg,score_fusion='max',normal_dwell={'minimum_complete_runs':10},dwell_context={'distribution':'entry'},dwell_score='fit_lognormal_cdf',dwell_lognormal={'sigma_floor':.05},transition_evidence_gate='consecutive_observed')
    def f(seed):
        d=fixture(seed);n=len(d['phases']);d.update(relation_valid=np.ones(n,bool),tracks=np.tile([10,20],n),relation_detection_indices=np.arange(n*2).reshape(n,2));d['object_frames']=np.repeat(np.arange(n),2);d['roles']=np.zeros(n*2,int);d['crop_features']=np.repeat(d['crop_features'],2,axis=0);return d
    old=LognormalDwellBaseline(cfg,t.process);old.fit([f(0),f(1)]);new=copy.deepcopy(old);new.cfg=dict(cfg,transition_evidence_gate='same_track_pair');cal=f(2);cal['tracks'][20::2]=30
    for m in [old,new]:m.calibrate([cal])
    for key in old.calibration:np.testing.assert_array_equal(old.calibration[key],new.calibration[key])
    for key in old.state_process_references:np.testing.assert_array_equal(old.state_process_references[key],new.state_process_references[key])
    np.testing.assert_array_equal(old.process_reference,new.process_reference)
    a,b=old.score(cal),new.score(cal)
    for key in ['visual','transition','transition_raw','dwell','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']:np.testing.assert_array_equal(a[key],b[key])
    mask=same_track_pair_transition_mask(cal);np.testing.assert_array_equal(b['transition_gated'],np.where(mask,a['transition'],0))
    np.testing.assert_array_equal(b['combined'],np.maximum(a['visual'],np.maximum(b['transition_gated'],np.where(a['dwell_valid'],a['dwell'],0))))
    assert np.all(b['combined']<=a['combined'])
    assert new.threshold==np.quantile(b['combined'],.99,method='higher')
