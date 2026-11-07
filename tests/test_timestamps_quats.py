import numpy as np
import pytest

from src.prep.cleaning import clean_session
from src.prep.timestamps import clean_timestamps


def test_exact_duplicate_is_dropped():
    t = np.array([0.0, 0.01, 0.01, 0.02])
    q = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    result = clean_timestamps(t, q)
    assert not result.rejected
    assert result.n_kept == 3
    assert result.removed_by_reason["exact_duplicate"] == 1


def test_conflicting_duplicate_rejects():
    t = np.array([0.0, 0.01, 0.01, 0.02])
    q = np.array(
        [
            [0, 0, 0, 1],
            [0, 0.1, 0, 0.995],
            [0, 0.2, 0, 0.98],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    result = clean_timestamps(t, q)
    assert result.rejected
    assert result.reject_reason == "duplicate_timestamp_conflicting_quaternion"


def test_non_monotonic_is_not_sorted():
    t = np.array([0.0, 0.02, 0.01, 0.03])
    q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (4, 1))
    result = clean_timestamps(t, q)
    assert result.rejected
    assert result.reject_reason == "non_monotonic_timestamps"
    assert np.allclose(result.timestamps, t)


def test_hemisphere_flip_and_unit_norm():
    q = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]], dtype=float)
    t = np.array([0.0, 0.01])
    cleaned = clean_session(t, q)
    assert not cleaned.rejected
    assert cleaned.n_hemisphere_flips == 1
    assert cleaned.unit_norm_ok
    assert np.dot(cleaned.quaternions[0], cleaned.quaternions[1]) > 0


def test_near_zero_quaternion_rejected():
    t = np.array([0.0, 0.01])
    q = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    cleaned = clean_session(t, q)
    assert cleaned.rejected
    assert cleaned.reject_reason == "near_zero_quaternion"


def test_fast_motion_is_kept():
    t = np.array([0.0, 0.01])
    q = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]])
    cleaned = clean_session(t, q)
    assert not cleaned.rejected
    assert cleaned.n_discontinuities >= 1
    assert len(cleaned.quaternions) == 2
