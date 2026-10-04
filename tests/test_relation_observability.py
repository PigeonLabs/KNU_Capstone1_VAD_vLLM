import numpy as np
from ipad_vad.relation_observability import candidate_masks,trace_role_candidates,select_boundary_cases


def test_exclusive_failure_categories_and_threshold_boundary():
    tests=[([],[],'no_raw_candidate'),([0,-1],[True,True],'no_positive_area'),([.1],[False],'semantic_excluded'),([.8],[True],'area_excluded'),([.1,.8],[False,True],'no_joint_candidate'),([.1,.8],[True,False],'candidate_available')]
    for areas,semantic,reason in tests:assert candidate_masks(np.array(areas),np.array(semantic,bool),.5)[3]==reason
    assert candidate_masks(np.array([.5+1e-8]),np.array([True]),.5)[2].all()


def test_prior_preference_filter_change_missing_and_target_has_no_semantic_gate():
    # Anchor prior 1 survives a lower-confidence frame, then is filtered at margin=0.
    frames=[0,0,1,1,1,2,2,2,3,4,4];roles=[1,0,1,1,0,1,1,0,0,1,0];tracks=[1,9,1,2,9,1,2,9,9,2,9]
    d={'indices':np.arange(5)*4,'object_frames':np.array(frames),'roles':np.array(roles),'tracks':np.array(tracks),'boxes':np.tile([.1,.1,.3,.3],(len(frames),1)),'confidence':np.array([1,1,.1,.9,1,1,.5,1,1,1,1.])}
    raw=np.ones(len(frames));gate=np.where(np.array(roles)==1,1.,-9);gate[5]=0
    rows,chosen,valid=trace_role_candidates(d,1,0,{1:.5,0:.5},raw,gate)
    np.testing.assert_array_equal(chosen[:,0],[0,2,6,-1,9]);np.testing.assert_array_equal(valid,[True,True,True,False,True])
    assert rows[1]['roles']['anchor']['selection_event']=='retained'
    assert rows[2]['roles']['anchor']['selection_event']=='prior_candidate_filtered'
    assert rows[3]['roles']['anchor']['reason']=='no_raw_candidate'
    assert rows[4]['roles']['anchor']['selection_event']=='retained'
    assert all(r['roles']['target']['semantic_raw_pass_count'] is None for r in rows)
    d['tracks'][9]=3
    assert trace_role_candidates(d,1,0,{1:.5,0:.5},raw,gate)[0][4]['roles']['anchor']['selection_event']=='prior_candidate_absent'


def test_case_selection_is_video_round_robin_and_preserves_membership():
    episodes=[]
    for seq in ['01','02','03']:
        for i in range(3):episodes.append({'sequence':seq,'partition':'fit','episode_id':i,'status':'unknown_entry','start_reason':'reacquired','end_reason':'relation_missing','start_sample':i+1,'first_observed_source_frame':4*(i+1)})
    a=select_boundary_cases(episodes,4);b=select_boundary_cases(list(reversed(episodes)),4)
    assert a==b and a['groups']['unknown_reacquired']==['01_0004','02_0004','03_0004','01_0008']
