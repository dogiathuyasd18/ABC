"""NumPy quaternion helpers. Input and stored order is xyzw."""

from __future__ import annotations

import numpy as np

from src.prep.config import NEAR_ZERO_NORM, UNIT_NORM_ATOL


def as_xyzw(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Invalid quaternion shape: {values.shape}; expected (N, 4)")
    if not np.all(np.isfinite(values)):
        raise ValueError("Quaternion array contains non-finite values")
    return values


def norms(quaternions: np.ndarray) -> np.ndarray:
    return np.linalg.norm(as_xyzw(quaternions), axis=1)


def normalize(
    quaternions: np.ndarray,
    near_zero: float = NEAR_ZERO_NORM,
) -> np.ndarray:
    values = as_xyzw(quaternions)
    if not np.isfinite(near_zero) or near_zero <= 0:
        raise ValueError("near_zero must be finite and positive")
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(lengths < near_zero):
        raise ValueError("Near-zero quaternion cannot be normalised")
    return values / lengths


def hemisphere_align(quaternions: np.ndarray) -> tuple[np.ndarray, int]:
    """Flip signs so adjacent unit quaternions occupy the same hemisphere."""
    aligned = normalize(quaternions)
    flips = 0
    for index in range(1, len(aligned)):
        if float(np.dot(aligned[index - 1], aligned[index])) < 0.0:
            aligned[index] *= -1.0
            flips += 1
    return aligned, flips


def geodesic_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize(a)
    b = normalize(b)
    if a.shape != b.shape:
        raise ValueError(f"Quaternion arrays must have equal shapes, got {a.shape} and {b.shape}")
    dots = np.clip(np.abs(np.sum(a * b, axis=1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def consecutive_geodesic_degrees(quaternions: np.ndarray) -> np.ndarray:
    values = as_xyzw(quaternions)
    if len(values) < 2:
        return np.zeros(0, dtype=np.float64)
    return geodesic_degrees(values[:-1], values[1:])


def unit_norm_ok(
    quaternions: np.ndarray,
    atol: float = UNIT_NORM_ATOL,
) -> bool:
    values = as_xyzw(quaternions)
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative")
    if len(values) == 0:
        return False
    return bool(np.allclose(norms(values), 1.0, rtol=0.0, atol=atol))

