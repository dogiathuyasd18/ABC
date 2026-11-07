"""Cleaning of PASS sessions: timestamps then quaternions.

Fast head motion is not deleted. Discontinuities are counted, not removed,
unless they are non-finite or near-zero quaternions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.prep.config import CleanConfig
from src.prep.quat_np import consecutive_geodesic_degrees, hemisphere_align, norms, normalize, unit_norm_ok
from src.prep.timestamps import TimestampCleanResult, clean_timestamps, interval_stats, stats_as_dict


@dataclass
class CleanResult:
    timestamps: np.ndarray
    quaternions: np.ndarray
    rejected: bool
    reject_reason: str
    n_input: int
    n_kept: int
    n_removed: int
    removed_by_reason: dict = field(default_factory=dict)
    n_hemisphere_flips: int = 0
    n_discontinuities: int = 0
    max_step_degrees: float = 0.0
    interval: dict = field(default_factory=dict)
    unit_norm_ok: bool = False


def clean_quaternions(quaternions: np.ndarray, cfg: CleanConfig | None = None) -> tuple[np.ndarray, dict, str]:
    cfg = cfg or CleanConfig()
    quaternions = np.asarray(quaternions, dtype=np.float64)
    tallies = {
        "non_finite_quaternion": 0,
        "near_zero_quaternion": 0,
    }
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        return quaternions, tallies, "invalid_quaternion_shape"
    if len(quaternions) == 0:
        return quaternions, tallies, "empty_quaternions"

    finite = np.isfinite(quaternions).all(axis=1)
    tallies["non_finite_quaternion"] = int((~finite).sum())
    if not finite.all():
        return quaternions, tallies, "non_finite_quaternions"

    lengths = norms(quaternions)
    near_zero = lengths < cfg.near_zero_norm
    tallies["near_zero_quaternion"] = int(near_zero.sum())
    if near_zero.any():
        return quaternions, tallies, "near_zero_quaternion"

    unit = normalize(quaternions, cfg.near_zero_norm)
    aligned, flips = hemisphere_align(unit)
    if not unit_norm_ok(aligned, cfg.unit_norm_atol):
        return aligned, tallies, "unit_norm_failed_after_normalize"
    tallies["hemisphere_flips"] = flips
    return aligned, tallies, ""


def clean_session(
    timestamps: np.ndarray,
    quaternions: np.ndarray,
    cfg: CleanConfig | None = None,
) -> CleanResult:
    cfg = cfg or CleanConfig()
    time_result: TimestampCleanResult = clean_timestamps(timestamps, quaternions, cfg)
    if time_result.rejected:
        return CleanResult(
            timestamps=time_result.timestamps,
            quaternions=time_result.quaternions,
            rejected=True,
            reject_reason=time_result.reject_reason,
            n_input=time_result.n_input,
            n_kept=time_result.n_kept,
            n_removed=time_result.n_removed,
            removed_by_reason=time_result.removed_by_reason,
            interval=stats_as_dict(interval_stats(time_result.timestamps)),
        )

    unit, quat_tallies, quat_reason = clean_quaternions(time_result.quaternions, cfg)
    removed = dict(time_result.removed_by_reason)
    removed.update({key: value for key, value in quat_tallies.items() if key != "hemisphere_flips"})
    if quat_reason:
        return CleanResult(
            timestamps=time_result.timestamps,
            quaternions=unit,
            rejected=True,
            reject_reason=quat_reason,
            n_input=time_result.n_input,
            n_kept=len(time_result.timestamps),
            n_removed=time_result.n_removed,
            removed_by_reason=removed,
            n_hemisphere_flips=int(quat_tallies.get("hemisphere_flips", 0)),
            interval=stats_as_dict(interval_stats(time_result.timestamps)),
        )

    steps = consecutive_geodesic_degrees(unit)
    n_disc = int(np.sum(steps > cfg.discontinuity_degrees)) if len(steps) else 0
    max_step = float(np.max(steps)) if len(steps) else 0.0
    # Fast motion is retained. The count is diagnostic only.
    return CleanResult(
        timestamps=time_result.timestamps.astype(np.float64),
        quaternions=unit.astype(np.float64),
        rejected=False,
        reject_reason="",
        n_input=time_result.n_input,
        n_kept=len(time_result.timestamps),
        n_removed=time_result.n_removed,
        removed_by_reason=removed,
        n_hemisphere_flips=int(quat_tallies.get("hemisphere_flips", 0)),
        n_discontinuities=n_disc,
        max_step_degrees=max_step,
        interval=stats_as_dict(interval_stats(time_result.timestamps)),
        unit_norm_ok=True,
    )
