
import numpy as np

from src.prep.gaps import gap_report, split_on_gaps
from src.prep.resampling import resample_session
from src.prep.timestamps import clean_timestamps, interval_stats
from src.prep.quat_np import normalize, hemisphere_align, geodesic_degrees



def test_quaternion_operations():
    q = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, -1.0],
    ])

    normalized = normalize(q)
    aligned, flips = hemisphere_align(normalized)

    assert normalized.shape == (2, 4)
    assert aligned.shape == (2, 4)

    assert np.allclose(
        np.linalg.norm(normalized, axis=1),
        1.0,
        atol=1e-7,
    )

    dots = np.sum(aligned[:-1] * aligned[1:], axis=1)

    assert np.all(dots >= 0.0)
    assert np.allclose(aligned[0], aligned[1], atol=1e-7)
    assert flips == 1



def test_gap_segmentation():
    timestamps = np.array([
        0.00, 0.01, 0.02,
        0.08, 0.09, 0.10,
    ])

    quaternions = np.tile(
        [0.0, 0.0, 0.0, 1.0],
        (len(timestamps), 1),
    )

    report = gap_report(timestamps)
    segments = split_on_gaps(timestamps, quaternions)

    assert report.n_split == 1
    assert len(segments) == 2
    assert [len(segment[0]) for segment in segments] == [3, 3]


def test_timestamp_validation():
    timestamps = np.arange(100, dtype=np.float64) / 100.0

    stats = interval_stats(timestamps)

    assert stats is not None


def test_resampling():
    timestamps = np.arange(101, dtype=np.float64) / 100.0

    angles = np.zeros((101, 3), dtype=np.float64)
    quaternions = np.tile(
        [0.0, 0.0, 0.0, 1.0],
        (101, 1),
    )

    result = resample_session(timestamps, quaternions)

    assert result is not None
    assert len(result.segments) > 0



from src.prep.windows import (
    WindowIndex,
    window_starts,
    index_segment,
    session_is_trainable,
    count_windows,
)


def test_window_indexing():
    starts = window_starts(
        n_samples=500,
        window=420,
        stride=20,
    )

    assert starts == [0, 20, 40, 60, 80]


def test_window_boundary():
    indices = index_segment(
        user_id="user_01",
        session_id="session_01",
        segment_id="0",
        n_samples=500,
    )

    for item in indices:
        assert item.end <= 500
        assert item.target_start == item.start + item.context
        assert item.window_samples == item.context + item.horizon


def test_window_exact_length():
    assert window_starts(420, 420, 20) == [0]
    assert window_starts(419, 420, 20) == []


def test_window_segment_isolation():
    lengths = [300, 300]

    assert session_is_trainable(lengths) is False
    assert count_windows(lengths) == 0


def test_window_invalid_input():
    import pytest

    with pytest.raises(ValueError):
        window_starts(-1, 420, 20)

    with pytest.raises(ValueError):
        window_starts(500, 0, 20)

    with pytest.raises(ValueError):
        window_starts(500, 420, 0)
