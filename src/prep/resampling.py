"""Independent 100 Hz SLERP resampling of continuous timestamp segments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from src.prep.config import CleanConfig, TARGET_SAMPLE_INTERVAL
from src.prep.gaps import GapReport, gap_report, report_as_dict, split_on_gaps
from src.prep.quat_np import hemisphere_align, normalize


@dataclass
class ResampledSegment:
    timestamps: np.ndarray
    quaternions: np.ndarray
    n_source: int
    n_resampled: int
    n_interpolated: int
    duration_s: float


@dataclass
class ResampleResult:
    segments: list[ResampledSegment]
    n_source: int
    n_resampled: int
    n_interpolated: int
    n_segments: int
    gap: dict
    rejected: bool = False
    reject_reason: str = ""


def _validate_inputs(
    timestamps: np.ndarray,
    quaternions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)

    if timestamps.ndim != 1:
        raise ValueError(f"timestamps must be 1D, got {timestamps.shape}")
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"quaternions must have shape (N, 4), got {quaternions.shape}")
    if len(timestamps) != len(quaternions):
        raise ValueError("timestamp_quaternion_length_mismatch")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps_contain_non_finite_values")
    if not np.all(np.isfinite(quaternions)):
        raise ValueError("quaternions_contain_non_finite_values")
    if len(timestamps) >= 2 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps_must_be_strictly_increasing")

    return timestamps, quaternions


def _count_non_source_grid_points(
    grid: np.ndarray,
    source_timestamps: np.ndarray,
    tolerance: float,
) -> int:
    """Count grid points not matching a source timestamp, without an N x M array."""
    right = np.searchsorted(source_timestamps, grid, side="left")
    left = np.clip(right - 1, 0, len(source_timestamps) - 1)
    right_clipped = np.clip(right, 0, len(source_timestamps) - 1)

    left_distance = np.abs(grid - source_timestamps[left])
    right_distance = np.abs(grid - source_timestamps[right_clipped])
    nearest_distance = np.minimum(left_distance, right_distance)
    return int(np.count_nonzero(nearest_distance > tolerance))


def _make_grid(
    start: float,
    end: float,
    interval: float,
    tolerance: float,
) -> np.ndarray:
    duration = end - start
    if not np.isfinite(duration) or duration < interval - tolerance:
        return np.empty(0, dtype=np.float64)

    nearest_steps = int(np.rint(duration / interval))
    lands_on_endpoint = (
        nearest_steps >= 1
        and abs(duration - nearest_steps * interval) <= tolerance
    )

    if lands_on_endpoint:
        n_intervals = nearest_steps
    else:
        n_intervals = int(np.floor((duration + tolerance) / interval))

    if n_intervals < 1:
        return np.empty(0, dtype=np.float64)

    grid = start + np.arange(n_intervals + 1, dtype=np.float64) * interval

    if lands_on_endpoint:
        grid[-1] = end

    grid = grid[grid <= end + tolerance]
    if len(grid) < 2:
        return np.empty(0, dtype=np.float64)

    if np.any(np.diff(grid) <= 0):
        raise ValueError("generated_resampling_grid_is_not_strictly_increasing")
    if grid[0] < start - tolerance or grid[-1] > end + tolerance:
        raise ValueError("generated_resampling_grid_exceeds_source_range")

    return grid


def _resample_one(
    timestamps: np.ndarray,
    quaternions: np.ndarray,
    cfg: CleanConfig,
) -> ResampledSegment | None:
    timestamps, quaternions = _validate_inputs(timestamps, quaternions)
    if len(timestamps) < 2:
        return None

    aligned = hemisphere_align(normalize(quaternions))[0]
    grid = _make_grid(
        float(timestamps[0]),
        float(timestamps[-1]),
        float(TARGET_SAMPLE_INTERVAL),
        float(cfg.timestamp_eps),
    )
    if len(grid) < 2:
        return None

    rotations = Rotation.from_quat(aligned)
    interpolator = Slerp(timestamps, rotations)
    interpolated = interpolator(grid).as_quat()
    interpolated = hemisphere_align(normalize(interpolated))[0]

    n_interpolated = _count_non_source_grid_points(
        grid,
        timestamps,
        float(cfg.timestamp_eps),
    )

    return ResampledSegment(
        timestamps=grid.astype(np.float64, copy=False),
        quaternions=interpolated.astype(np.float32, copy=False),
        n_source=len(timestamps),
        n_resampled=len(grid),
        n_interpolated=n_interpolated,
        duration_s=float(grid[-1] - grid[0]),
    )


def resample_session(
    timestamps: np.ndarray,
    quaternions: np.ndarray,
    cfg: CleanConfig | None = None,
) -> ResampleResult:
    cfg = cfg or CleanConfig()
    try:
        timestamps, quaternions = _validate_inputs(timestamps, quaternions)
        if len(timestamps) < 2:
            raise ValueError("too_few_samples_for_resampling")

        gaps: GapReport = gap_report(timestamps, cfg)
        pieces = split_on_gaps(timestamps, quaternions, cfg)

        segments: list[ResampledSegment] = []
        for piece_t, piece_q in pieces:
            resampled = _resample_one(piece_t, piece_q, cfg)
            if resampled is not None:
                segments.append(resampled)

        return ResampleResult(
            segments=segments,
            n_source=len(timestamps),
            n_resampled=sum(segment.n_resampled for segment in segments),
            n_interpolated=sum(segment.n_interpolated for segment in segments),
            n_segments=len(segments),
            gap=report_as_dict(gaps),
            rejected=False,
            reject_reason="",
        )
    except (ValueError, TypeError, FloatingPointError) as exc:
        return ResampleResult(
            segments=[],
            n_source=len(timestamps) if np.ndim(timestamps) > 0 else 0,
            n_resampled=0,
            n_interpolated=0,
            n_segments=0,
            gap={},
            rejected=True,
            reject_reason=str(exc),
        )
