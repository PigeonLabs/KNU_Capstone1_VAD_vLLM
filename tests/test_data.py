import numpy as np
import pytest
from ipad_vad.data import frames_in_order, evaluation_labels, bounded_offset_consensus, hold_scores


def test_mismatch_cannot_silently_enter_metrics():
    assert np.all(evaluation_labels([0,1,0], 4) == -1)
    assert np.all(evaluation_labels(np.ones(676), 626) == -1)
    assert evaluation_labels([0,1],2).tolist() == [0,1]


def test_reject_invalid_annotation():
    for labels in [[0,np.nan], [[0,1]], [0,2]]:
        with pytest.raises(ValueError): evaluation_labels(labels,2)


def test_numeric_order_crosses_thousand(tmp_path):
    for i in range(1002): (tmp_path / f"{i:03}.jpg").touch()
    assert frames_in_order(tmp_path)[1000].name == "1000.jpg"


def test_gap_rejected(tmp_path):
    (tmp_path/"000.jpg").touch();(tmp_path/"002.jpg").touch()
    with pytest.raises(ValueError): frames_in_order(tmp_path)


def test_secondary_offset_mask_is_not_exact_alignment():
    labels, _ = bounded_offset_consensus([0,0,0,1,1],6)
    assert labels.tolist() == [-1,0,-1,-1,-1,-1]
    with pytest.raises(ValueError): bounded_offset_consensus([0]*676,626)


def test_dense_scores_never_use_future_samples():
    assert hold_scores([0,4],[.1,.9],7).tolist() == [.1,.1,.1,.1,.9,.9,.9]
