from ipad_vad.events import anomaly_events,summarize_events


def test_missed_events_remain_in_denominator_and_delay_is_null():
    e=anomaly_events([0,1,1,0,1,1,1,0],[0,0,0,0,0,0,1,0])
    assert len(e)==2 and e[0]['first_alarm_delay_frames'] is None
    assert e[1]['first_alarm_delay_frames']==2
    s=summarize_events(e)
    assert s['event_coverage']==.5 and s['missed_events']==1
    assert s['median_delay_detected_only_frames']==2


def test_preexisting_alarm_and_censoring_are_visible():
    e=anomaly_events([1,1,-1,1,0,1,1],[1,1,0,1,1,1,1])
    assert len(e)==3
    assert e[0]['left_boundary_censored'] and e[0]['right_boundary_censored']
    assert e[1]['left_boundary_censored']
    assert e[2]['alarm_active_before_onset'] and e[2]['first_alarm_delay_frames']==0
    assert e[2]['right_boundary_censored']
