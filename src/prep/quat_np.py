"""NumPy quaternion helpers for preprocessing. Stored order is xyzw."""

from __future__ import annotations

import numpy as np

from src.prep.config import NEAR_ZERO_NORM, UNIT_NORM_ATOL


def as_xyzw(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"Invalid quaternion shape: {quaternions.shape}")
    return quaternions


def norms(quaternions: np.ndarray) -> np.ndarray:
    return np.linalg.norm(as_xyzw(quaternions), axis=1)


def geodesic_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = as_xyzw(a)
    b = as_xyzw(b)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), NEAR_ZERO_NORM, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), NEAR_ZERO_NORM, None)
    dots = np.clip(np.abs(np.sum(a * b, axis=1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def hemisphere_align(quaternions: np.ndarray) -> tuple[np.ndarray, int]:
    aligned = np.array(as_xyzw(quaternions), copy=True)
    flips = 0
    for index in range(1, len(aligned)):
        if np.dot(aligned[index - 1], aligned[index]) < 0.0:
            aligned[index] *= -1.0
            flips += 1
    return aligned, flips


def normalize(quaternions: np.ndarray, near_zero: float = NEAR_ZERO_NORM) -> np.ndarray:
    values = as_xyzw(quaternions)
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(lengths < near_zero):
        raise ValueError("Near-zero quaternion cannot be normalised")
    return values / lengths


def consecutive_geodesic_degrees(quaternions: np.ndarray) -> np.ndarray:
    if len(quaternions) < 2:
        return np.zeros(0, dtype=np.float64)
    return geodesic_degrees(quaternions[:-1], quaternions[1:])


def unit_norm_ok(quaternions: np.ndarray, atol: float = UNIT_NORM_ATOL) -> bool:
    return bool(np.allclose(norms(quaternions), 1.0, atol=atol))
