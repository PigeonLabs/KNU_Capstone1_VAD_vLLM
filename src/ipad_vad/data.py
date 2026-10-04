"""Frame-index timebase and conservative annotation handling.

Unknown labels affect evaluation only, never feature extraction or scoring.
No seconds, cycle boundaries, or corrected ground truth are fabricated.
"""
from pathlib import Path
import numpy as np


def frames_in_order(directory):
    files = list(Path(directory).glob("*.jpg"))
    if not files:
        raise ValueError(f"No JPG frames: {directory}")
    files.sort(key=lambda p: int(p.stem))
    ids = [int(p.stem) for p in files]
    if len(set(ids)) != len(ids) or ids != list(range(len(ids))):
        raise ValueError(f"Non-contiguous or nonzero-based frame IDs: {directory}")
    return files


def validate_labels(labels):
    a = np.asarray(labels)
    if a.ndim != 1 or not np.isin(a, [0, 1]).all():
        raise ValueError("Expected a finite, binary, 1-D annotation")
    return a.astype(np.int8)


def evaluation_labels(labels, frame_count):
    a = validate_labels(labels)
    if len(a) != frame_count:
        return np.full(frame_count, -1, dtype=np.int8)
    return a


def bounded_offset_consensus(labels, frame_count):
    """SECONDARY sensitivity only. Assumes the true offset remains within ±1."""
    a = validate_labels(labels)
    if abs(len(a) - frame_count) != 1:
        raise ValueError("Only defined for ±1 length discrepancies")
    candidates = np.full((3, frame_count), -1, dtype=np.int8)
    for row, offset in enumerate((-1, 0, 1)):
        ids = np.arange(frame_count) + offset
        valid = (ids >= 0) & (ids < len(a))
        candidates[row, valid] = a[ids[valid]]
    known = (candidates >= 0).all(0) & (candidates == candidates[0]).all(0)
    return np.where(known, candidates[0], -1), candidates


def sample_indices(frame_count, stride):
    if frame_count < 1 or stride < 1:
        raise ValueError("Positive frame_count and stride required")
    return np.arange(0, frame_count, stride, dtype=np.int64)


def hold_scores(indices, scores, frame_count):
    """Causal zero-order hold. Never interpolate from a future sampled frame."""
    indices, scores = np.asarray(indices), np.asarray(scores)
    if len(indices) != len(scores) or not len(indices) or indices[0] != 0:
        raise ValueError("Aligned samples starting at frame 0 required")
    if np.any(np.diff(indices) <= 0) or indices[-1] >= frame_count:
        raise ValueError("Invalid sample indices")
    return scores[np.searchsorted(indices, np.arange(frame_count), side="right") - 1]
