import numpy as np
import pytest
from ipad_vad.transition_trace import transition_rows,coverage,select_cases


def test_coverage_excludes_boundaries_and_counts_distinct_videos():
    data={'phases':np.array([3,2,3]),'indices':np.array([0,4,8]),'relation_valid':np.array([True,True,False])}
    frames=[{'selected':{'anchor':{'track':1},'target':None}}]*3
    rows=transition_rows('01','fit',data,frames)+transition_rows('02','calibration',data,frames)
    c=coverage(rows);assert sum(map(sum,c['all']['counts']))==4;assert c['observed']['counts'][3][2]==2;assert c['observed']['distinct_videos'][3][2]==2;assert rows[0]['anchor_track_changed'] is False and rows[0]['target_track_changed'] is None
    with pytest.raises(ValueError):transition_rows('01','fit',dict(data,relation_valid=np.ones(3)),frames)


def test_selection_preserves_failures_and_balances_videos_without_future_information():
    def row(seq,frame,dest=2):return {'case_id':f'{seq}_{frame}','sequence':seq,'partition':'calibration' if seq=='02' else 'fit','source_frame':frame,'observed_pair':True,'previous_phase':3,'current_phase':dest}
    rows=[row('02',152),row('02',376),row('01',4),row('01',8),row('03',4),row('01',12,3),row('03',8,1)]
    selected=select_cases(rows,4);targets=[r['case_id'] for r in selected if r['selection_group']=='target_3_to_2'];assert targets==['02_152','02_376','01_4','03_4'];assert len({r['case_id'] for r in selected})==len(selected)
    assert select_cases(list(reversed(rows)),4)==selected
    with pytest.raises(ValueError):select_cases(rows[1:],4)
