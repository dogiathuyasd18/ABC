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
    if not np.isfinite(value) or not np.isfinite(threshold):
        return False
    if mode == "greater_than":
        return bool(value > threshold)
    if mode == "greater_or_equal":
        return bool(value >= threshold)
    raise ValueError(f"Unknown comparison: {mode}")


def to_euler_xyz_degrees(
    quaternions: np.ndarray,
    euler_order: str = "xyz",
) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("quaternions must have shape (N, 4)")
    if len(quaternions) == 0:
        raise ValueError("quaternions are empty")
    if not np.all(np.isfinite(quaternions)):
        raise ValueError("Non-finite quaternion values")

    rotation = Rotation.from_quat(quaternions)
    angles = rotation.as_euler(euler_order, degrees=True)
    if not np.all(np.isfinite(angles)):
        raise ValueError("Non-finite Euler angles")
    return angles


def _unwrap_euler_degrees(euler_angles: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(euler_angles)
    return np.rad2deg(np.unwrap(radians, axis=0))


def analyze_motion_windows(
    timestamps: np.ndarray,
    euler_angles: np.ndarray,
    cfg: FilterConfig | None = None,
) -> list[dict]:
    """Compute motion statistics in complete, non-overlapping windows."""
    cfg = cfg or FilterConfig()
    timestamps = np.asarray(timestamps, dtype=np.float64)
    euler_angles = np.asarray(euler_angles, dtype=np.float64)

    if timestamps.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if euler_angles.ndim != 2 or euler_angles.shape != (len(timestamps), 3):
        raise ValueError("euler_angles must have shape (N, 3), matching timestamps")
    if not np.all(np.isfinite(timestamps)) or not np.all(np.isfinite(euler_angles)):
        raise ValueError("timestamps and Euler angles must be finite")
    if len(timestamps) < 2:
        return []
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    continuous_angles = _unwrap_euler_degrees(euler_angles)
    window_results = []
    start_time = float(timestamps[0])
    end_time = float(timestamps[-1])
    window_size = float(cfg.motion_window_seconds)
    window_start = start_time
    eps = max(1e-9, window_size * 1e-9)

    while window_start + window_size <= end_time + eps:
        window_end = window_start + window_size
        sample_mask = (timestamps >= window_start) & (timestamps < window_end)
        window_times = timestamps[sample_mask]
        window_angles = continuous_angles[sample_mask]

        if len(window_times) >= cfg.min_valid_samples_per_window:
            interval_mask = (
                (timestamps[:-1] >= window_start)
                & (timestamps[1:] < window_end)
            )
            interval_durations = np.diff(timestamps)[interval_mask]
            previous_angles = continuous_angles[:-1][interval_mask]
            next_angles = continuous_angles[1:][interval_mask]

            if len(interval_durations) > 0:
                sweep = (
                    np.percentile(window_angles, 90, axis=0)
                    - np.percentile(window_angles, 10, axis=0)
                )
                raw_differences = next_angles - previous_angles
                angle_diffs = np.abs(
                    (raw_differences + 180.0) % 360.0 - 180.0
                )
                total_axis_motion = np.sum(angle_diffs, axis=0)
                velocity = total_axis_motion / window_size

                window_results.append({
                    "sweep": sweep,
                    "velocity": velocity,
                    "num_samples": int(len(window_times)),
                    "num_intervals": int(len(interval_durations)),
                    "duration_seconds": window_size,
                })

        window_start += window_size

    return window_results


def aggregate_motion(
    window_results: list[dict],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if not window_results:
        raise ValueError("Cannot aggregate an empty list of motion windows")

    sweeps = np.asarray([row["sweep"] for row in window_results], dtype=np.float64)
    velocities = np.asarray([row["velocity"] for row in window_results], dtype=np.float64)

    if sweeps.ndim != 2 or sweeps.shape[1] != 3:
        raise ValueError("Motion-window sweep values must have shape (N, 3)")
    if velocities.shape != sweeps.shape:
        raise ValueError("Motion-window velocity values must match sweep shape")
    if not np.all(np.isfinite(sweeps)) or not np.all(np.isfinite(velocities)):
        raise ValueError("Motion-window statistics contain non-finite values")

    median_sweep = np.median(sweeps, axis=0)
    median_velocity = np.median(velocities, axis=0)
    return (
        median_sweep,
        median_velocity,
        float(np.max(median_sweep)),
        float(np.max(median_velocity)),
    )


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
    n_raw = len(timestamps) if timestamps.ndim > 0 else 0

    def fail(reason: str, **extra) -> FilterResult:
        values = dict(
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
        values.update(extra)
        return FilterResult(**values)

    if timestamps.ndim != 1:
        return fail("timestamps_must_be_1d")
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        return fail("quaternions_must_have_shape_N_by_4")
    if len(timestamps) != len(quaternions):
        return fail("timestamp_quaternion_length_mismatch")
    if n_raw == 0:
        return fail("empty_session")
    if not np.all(np.isfinite(timestamps)):
        return fail("non_finite_timestamps")
    if not np.all(np.isfinite(quaternions)):
        return fail("non_finite_quaternions")

    try:
        trimmed_t, trimmed_q, n_trim_removed = trim_startup(
            timestamps, quaternions, cfg.trim_seconds
        )
    except Exception as exc:
        return fail(f"trim_error:{type(exc).__name__}:{exc}")

    trimmed_t = np.asarray(trimmed_t, dtype=np.float64)
    trimmed_q = np.asarray(trimmed_q, dtype=np.float64)
    n_after_trim = len(trimmed_t)

    if trimmed_t.ndim != 1:
        return fail("trimmed_timestamps_must_be_1d", n_trim_removed=n_trim_removed)
    if trimmed_q.ndim != 2 or trimmed_q.shape[1] != 4:
        return fail("trimmed_quaternions_must_have_shape_N_by_4",
                    n_after_trim=n_after_trim, n_trim_removed=n_trim_removed)
    if len(trimmed_t) != len(trimmed_q):
        return fail("trimmed_array_length_mismatch",
                    n_after_trim=n_after_trim, n_trim_removed=n_trim_removed)
    if n_after_trim < cfg.min_samples_after_trim:
        return fail(
            "too_few_samples_after_trim",
            n_after_trim=n_after_trim,
            n_trim_removed=n_trim_removed,
        )
    if not np.all(np.isfinite(trimmed_t)) or not np.all(np.isfinite(trimmed_q)):
        return fail(
            "non_finite_values_after_trim",
            n_after_trim=n_after_trim,
            n_trim_removed=n_trim_removed,
        )

    try:
        stats = interval_stats(trimmed_t, cfg)
        interval_data = stats_as_dict(stats)
    except Exception as exc:
        return fail(
            f"timestamp_analysis_error:{type(exc).__name__}:{exc}",
            n_after_trim=n_after_trim,
            n_trim_removed=n_trim_removed,
        )

    extra = {
        "sampling_rate_hz": float(stats.sampling_rate_hz),
        "sampling_rate_valid": bool(stats.sampling_rate_valid),
        "n_after_trim": n_after_trim,
        "n_trim_removed": int(n_trim_removed),
        "interval": interval_data,
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
    except Exception as exc:
        return fail(
            f"quaternion_or_euler_invalid:{type(exc).__name__}:{exc}",
            **extra,
        )

    try:
        windows = analyze_motion_windows(trimmed_t, euler, cfg)
    except Exception as exc:
        return fail(
            f"motion_window_analysis_error:{type(exc).__name__}:{exc}",
            **extra,
        )

    if not windows:
        return fail("no_valid_motion_windows", **extra, n_valid_windows=0)

    try:
        median_sweep, median_velocity, session_sweep, session_velocity = (
            aggregate_motion(windows)
        )
    except Exception as exc:
        return fail(
            f"motion_aggregation_error:{type(exc).__name__}:{exc}",
            **extra,
            n_valid_windows=len(windows),
        )

    sweep_valid = _compare(
        session_sweep, cfg.min_sweep_degrees, cfg.sweep_comparison
    )
    velocity_valid = _compare(
        session_velocity, cfg.min_velocity_deg_s, cfg.velocity_comparison
    )

    extra.update(
        n_valid_windows=len(windows),
        sweep_xyz=tuple(float(v) for v in median_sweep),
        velocity_xyz=tuple(float(v) for v in median_velocity),
        session_sweep=session_sweep,
        session_velocity=session_velocity,
        sweep_valid=sweep_valid,
        velocity_valid=velocity_valid,
        timestamps_trim=trimmed_t,
        quaternions_trim=unit_q.astype(np.float64, copy=False),
    )

    if not sweep_valid:
        return fail("sweep_below_threshold", **extra)
    if not velocity_valid:
        return fail("velocity_below_threshold", **extra)

    extra.update(
        status="PASS",
        reject_reason="",
        spec_version=cfg.spec_version,
        n_raw=n_raw,
    )
    return FilterResult(**extra)


def filter_row(
    result: FilterResult,
    user_id: str,
    session_id: str,
    member_name: str,
    archive: str,
) -> dict:
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
