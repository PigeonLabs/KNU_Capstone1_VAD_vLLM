import copy
import numpy as np
import pytest
from ipad_vad.dwell_episodes import extract_dwell_episodes


def fixture(phases,pairs=None,valid=None,indices=None):
    n=len(phases);pairs=np.tile([10,20],(n,1)) if pairs is None else np.asarray(pairs,int)
    valid=np.ones(n,bool) if valid is None else np.asarray(valid,bool)
    selected=np.arange(2*n).reshape(n,2);selected[~valid]=-1
    indices=np.arange(n)*4 if indices is None else np.asarray(indices,int)
    return {'phases':np.array(phases,int),'indices':indices,'relation_valid':valid,'relation_detection_indices':selected,'tracks':pairs.ravel(),'object_frames':np.repeat(np.arange(n),2),'frame_count':np.array(int(indices[-1])+3 if n else 0)}


def prefix(d,n):
    use=d['object_frames']<n
    return {k:v[use] if k in ['tracks','object_frames'] else v[:n] if k!='frame_count' else v for k,v in d.items()}


def test_complete_exit_belongs_to_next_episode_and_terminal_lower_bound():
    d=fixture([0,1,1,2,2]);r=extract_dwell_episodes(d);a,b,c=r['episodes']
    assert a['status']=='unknown_entry' and a['exit_observed'] and a['duration_frames'] is None
    assert (b['status'],b['entry_context'],b['duration_frames'],b['observed_samples'])==('complete',0,8,2)
    assert (b['last_observed_source_frame'],b['boundary_source_frame'])==(8,12)
    assert (c['status'],c['entry_context'],c['censor_lower_bound_frames'],c['boundary_source_frame'])==('right_censored',1,4,None)
    np.testing.assert_array_equal(r['sample_episode_ids'],[0,1,1,2,2]);np.testing.assert_array_equal(r['sample_age_frames'],[-1,0,4,0,4])


@pytest.mark.parametrize('pair,reason',[([11,20],'anchor_changed'),([10,21],'target_changed'),([11,21],'both_changed')])
def test_track_change_censors_and_does_not_create_entry(pair,reason):
    d=fixture([0,1,1,2,2],[[10,20]]*3+[pair]*2);r=extract_dwell_episodes(d);old,new=r['episodes'][1:]
    assert old['status']=='right_censored' and old['end_reason']==reason and old['censor_lower_bound_frames']==4
    assert new['status']=='unknown_entry' and new['entry_context'] is None and new['start_reason']==reason
    assert new['duration_frames'] is None and new['censor_lower_bound_frames'] is None


def test_missing_reacquisition_unknown_and_zero_censor_bound():
    d=fixture([0,1,1,1,2,2],valid=[1,1,0,1,1,0],indices=[0,4,12,20,28,40]);r=extract_dwell_episodes(d);rows=r['episodes']
    assert rows[1]['end_reason']=='relation_missing' and rows[1]['censor_lower_bound_frames']==0
    assert rows[1]['boundary_source_frame']==12 and rows[1]['last_observed_source_frame']==4
    assert rows[2]['start_reason']=='reacquired' and not rows[2]['entry_observed']
    assert rows[3]['status']=='right_censored' and rows[3]['censor_lower_bound_frames']==0
    np.testing.assert_array_equal(r['sample_episode_ids'],[0,1,-1,2,3,-1]);np.testing.assert_array_equal(r['missing_sample_indices'],[2,5])


def test_no_entry_is_invented_at_video_start_after_initial_missing_or_empty():
    assert extract_dwell_episodes(fixture([]))['episodes']==[]
    assert extract_dwell_episodes(fixture([0,0],valid=[0,0]))['episodes']==[]
    r=extract_dwell_episodes(fixture([0,1,1],valid=[0,1,1]));assert len(r['episodes'])==1
    assert r['episodes'][0]['status']=='unknown_entry' and r['episodes'][0]['start_reason']=='reacquired'
    assert np.all(r['sample_age_frames']==-1)


def test_prefix_preserves_past_observation_state_but_not_future_completion():
    d=fixture([0,1,1,2,2,2,3],valid=[1,1,1,1,0,1,1]);full=extract_dwell_episodes(d)
    for n in range(len(d['phases'])+1):
        r=extract_dwell_episodes(prefix(d,n))
        for key in ['sample_episode_ids','sample_age_frames','sample_entry_context']:np.testing.assert_array_equal(r[key],full[key][:n])
        for row in r['episodes']:
            original=full['episodes'][row['episode_id']]
            for key in ['start_sample','first_observed_source_frame','phase','track_pair','start_reason','entry_observed','entry_context','entry_source_frame']:assert row[key]==original[key]
            if row['end_reason']!='video_end':assert row==original
    partial=extract_dwell_episodes(prefix(d,3))['episodes'][-1]
    assert partial['status']=='right_censored' and full['episodes'][1]['status']=='complete'


@pytest.mark.parametrize('key,value',[('indices',np.array([0,0])),('indices',np.array([-1,4])),('indices',np.array([0.,4.])),('phases',np.array([0.,1.])),('phases',np.array([0,-1])),('frame_count',np.array(4))])
def test_invalid_time_and_phase_metadata(key,value):
    d=fixture([0,1]);d[key]=value
    with pytest.raises(ValueError):extract_dwell_episodes(d)
