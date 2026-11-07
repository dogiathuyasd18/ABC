"""Session filtering for the 9th ABC Challenge data page.

PASS/FAIL here is only sampling rate, sweep, and angular velocity.
A session is not discarded for having fewer than 420 samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from src.prep.config import FilterConfig
from src.prep.quat_np import hemisphere_align, normalize
from src.prep.timestamps import interval_stats, stats_as_dict, trim_startup


def _compare(value: float, threshold: float, mode: str) -> bool:
    if mode == "greater_than":
        return value > threshold
    if mode == "greater_or_equal":
        return value >= threshold
    raise ValueError(f"Unknown comparison: {mode}")


def to_euler_xyz_degrees(quaternions: np.ndarray, euler_order: str = "xyz") -> np.ndarray:
    rotation = Rotation.from_quat(quaternions)
    angles = rotation.as_euler(euler_order, degrees=True)
    if not np.all(np.isfinite(angles)):
        raise ValueError("Non-finite Euler angles")
    return angles


def analyze_motion_windows(
    timestamps: np.ndarray,
    euler_angles: np.ndarray,
    cfg: FilterConfig | None = None,
) -> list[dict]:
    """Official tutorial: non-overlapping 5 s windows, skip < 10 samples."""
    cfg = cfg or FilterConfig()
    window_results = []
    if len(timestamps) == 0:
        return window_results

    start_time = float(timestamps[0])
    end_time = float(timestamps[-1])
    window_start = start_time
    window_size = cfg.motion_window_seconds

    while window_start + window_size <= end_time:
        window_end = window_start + window_size
        sample_mask = (timestamps >= window_start) & (timestamps < window_end)
        window_times = timestamps[sample_mask]
        window_angles = euler_angles[sample_mask]

        if len(window_times) < cfg.min_valid_samples_per_window:
            window_start += window_size
            continue

        sweep = np.percentile(window_angles, 90, axis=0) - np.percentile(window_angles, 10, axis=0)

        interval_mask = (timestamps[:-1] >= window_start) & (timestamps[1:] < window_end)
        previous_angles = euler_angles[:-1][interval_mask]
        next_angles = euler_angles[1:][interval_mask]
        if len(previous_angles) == 0:
            window_start += window_size
            continue

        angle_diffs = np.abs(next_angles - previous_angles)
        wrapped = np.where(angle_diffs > 180.0, 360.0 - angle_diffs, angle_diffs)
        velocity = np.sum(wrapped, axis=0) / window_size

        window_results.append(
            {
                "sweep": sweep,
                "velocity": velocity,
                "num_samples": len(window_times),
                "num_intervals": len(previous_angles),
            }
        )
        window_start += window_size

    return window_results


def aggregate_motion(window_results: list[dict]) -> tuple[np.ndarray, np.ndarray, float, float]:
    sweeps = np.array([row["sweep"] for row in window_results])
    velocities = np.array([row["velocity"] for row in window_results])
    median_sweep = np.median(sweeps, axis=0)
    median_velocity = np.median(velocities, axis=0)
    return median_sweep, median_velocity, float(np.max(median_sweep)), float(np.max(median_velocity))


@dataclass
class FilterResult:
    status: str
    reject_reason: str
    spec_version: str
    sampling_rate_hz: float
    sampling_rate_valid: bool
    n_raw: int
    n_after_trim: int
    n_trim_removed: int
    n_valid_windows: int
    sweep_xyz: tuple[float, float, float]
    velocity_xyz: tuple[float, float, float]
    session_sweep: float
    session_velocity: float
    sweep_valid: bool
    velocity_valid: bool
    interval: dict = field(default_factory=dict)
    timestamps_trim: np.ndarray | None = None
    quaternions_trim: np.ndarray | None = None

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def filter_session(
    timestamps: np.ndarray,
    quaternions: np.ndarray,
    cfg: FilterConfig | None = None,
) -> FilterResult:
    cfg = cfg or FilterConfig()
    timestamps = np.asarray(timestamps, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    n_raw = len(timestamps)

    def fail(reason: str, **extra) -> FilterResult:
        defaults = dict(
            status="FAIL",
            reject_reason=reason,
            spec_version=cfg.spec_version,
            sampling_rate_hz=float("nan"),
            sampling_rate_valid=False,
            n_raw=n_raw,
            n_after_trim=0,
            n_trim_removed=0,
            n_valid_windows=0,
            sweep_xyz=(float("nan"),) * 3,
            velocity_xyz=(float("nan"),) * 3,
            session_sweep=float("nan"),
            session_velocity=float("nan"),
            sweep_valid=False,
            velocity_valid=False,
            interval={},
        )
        defaults.update(extra)
        return FilterResult(**defaults)

    if n_raw == 0:
        return fail("empty_session")
    if not np.all(np.isfinite(timestamps)):
        return fail("non_finite_timestamps")
    if not np.all(np.isfinite(quaternions)):
        return fail("non_finite_quaternions")

    trimmed_t, trimmed_q, n_trim_removed = trim_startup(timestamps, quaternions, cfg.trim_seconds)
    n_after_trim = len(trimmed_t)
    if n_after_trim < cfg.min_samples_after_trim:
        return fail(
            "too_few_samples_after_trim",
            n_after_trim=n_after_trim,
            n_trim_removed=n_trim_removed,
        )

    stats = interval_stats(trimmed_t, cfg)
    extra = {
        "sampling_rate_hz": stats.sampling_rate_hz,
        "sampling_rate_valid": stats.sampling_rate_valid,
        "n_after_trim": n_after_trim,
        "n_trim_removed": n_trim_removed,
        "interval": stats_as_dict(stats),
    }
    if stats.n_non_monotonic:
        return fail("non_monotonic_timestamps", **extra)
    if stats.n_duplicate:
        return fail("duplicate_timestamps", **extra)
    if stats.suspected_units == "milliseconds_suspect":
        return fail("timestamp_units_not_seconds", **extra)
    if not stats.sampling_rate_valid:
        return fail("sampling_rate_out_of_range", **extra)

    try:
        unit_q = normalize(trimmed_q)
        unit_q, _flips = hemisphere_align(unit_q)
        euler = to_euler_xyz_degrees(unit_q, cfg.euler_order)
    except Exception:
        return fail("quaternion_or_euler_invalid", **extra)

    windows = analyze_motion_windows(trimmed_t, euler, cfg)
    if not windows:
        return fail("no_valid_motion_windows", **extra, n_valid_windows=0)

    median_sweep, median_velocity, session_sweep, session_velocity = aggregate_motion(windows)
    sweep_valid = _compare(session_sweep, cfg.min_sweep_degrees, cfg.sweep_comparison)
    velocity_valid = _compare(session_velocity, cfg.min_velocity_deg_s, cfg.velocity_comparison)

    extra.update(
        n_valid_windows=len(windows),
        sweep_xyz=tuple(float(v) for v in median_sweep),
        velocity_xyz=tuple(float(v) for v in median_velocity),
        session_sweep=session_sweep,
        session_velocity=session_velocity,
        sweep_valid=sweep_valid,
        velocity_valid=velocity_valid,
        timestamps_trim=trimmed_t,
        quaternions_trim=unit_q.astype(np.float64),
    )
    if not sweep_valid:
        return fail("sweep_below_threshold", **extra)
    if not velocity_valid:
        return fail("velocity_below_threshold", **extra)

    extra["status"] = "PASS"
    extra["reject_reason"] = ""
    extra["spec_version"] = cfg.spec_version
    extra["n_raw"] = n_raw
    return FilterResult(**extra)


def filter_row(result: FilterResult, user_id: str, session_id: str, member_name: str, archive: str) -> dict:
    sweep = result.sweep_xyz
    velocity = result.velocity_xyz
    return {
        "archive": archive,
        "user_id": user_id,
        "session_id": session_id,
        "member_name": member_name,
        "spec_version": result.spec_version,
        "status": result.status,
        "reject_reason": result.reject_reason,
        "frames_raw": result.n_raw,
        "frames_after_trim": result.n_after_trim,
        "sampling_rate": result.sampling_rate_hz,
        "sampling_rate_valid": result.sampling_rate_valid,
        "num_valid_windows": result.n_valid_windows,
        "sweep_x": sweep[0],
        "sweep_y": sweep[1],
        "sweep_z": sweep[2],
        "velocity_x": velocity[0],
        "velocity_y": velocity[1],
        "velocity_z": velocity[2],
        "session_sweep": result.session_sweep,
        "session_velocity": result.session_velocity,
        "sweep_valid": result.sweep_valid,
        "velocity_valid": result.velocity_valid,
        "session_valid": result.passed,
        **{f"interval_{key}": value for key, value in result.interval.items()},
    }
