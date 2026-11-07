"""Independent 100 Hz resampling of each continuous segment.

SLERP stays inside a segment. Grid points never leave the observed time
range. Segments that were split on a long gap are never joined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from src.prep.config import CleanConfig, TARGET_SAMPLE_INTERVAL, TARGET_SAMPLE_RATE_HZ
from src.prep.gaps import GapReport, gap_report, report_as_dict, split_on_gaps
from src.prep.quat_np import hemisphere_align


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


def _resample_one(
    timestamps: np.ndarray,
    quaternions: np.ndarray,
    cfg: CleanConfig,
) -> ResampledSegment | None:
    if len(timestamps) < 2:
        return None
    keep = np.ones(len(timestamps), dtype=bool)
    keep[1:] = np.diff(timestamps) > cfg.timestamp_eps
    timestamps = timestamps[keep]
    quaternions = quaternions[keep]
    if len(timestamps) < 2:
        return None

    duration = float(timestamps[-1] - timestamps[0])
    if duration < TARGET_SAMPLE_INTERVAL:
        return None

    grid = np.arange(timestamps[0], timestamps[-1], TARGET_SAMPLE_INTERVAL, dtype=np.float64)
    if len(grid) == 0:
        return None
    # Inclusive last stamp only when it lands on the grid; never beyond.
    if np.isclose(grid[-1] + TARGET_SAMPLE_INTERVAL, timestamps[-1], atol=cfg.timestamp_eps):
        grid = np.append(grid, timestamps[-1])
    grid = grid[grid <= timestamps[-1] + cfg.timestamp_eps]
    if len(grid) < 2:
        return None

    aligned, _flips = hemisphere_align(quaternions)
    interpolated = Slerp(timestamps, Rotation.from_quat(aligned))(grid).as_quat()
    interpolated, _flips = hemisphere_align(interpolated)

    nearest = np.min(np.abs(grid[:, None] - timestamps[None, :]), axis=1)
    n_interpolated = int(np.sum(nearest > cfg.timestamp_eps))
    return ResampledSegment(
        timestamps=grid.astype(np.float64),
        quaternions=interpolated.astype(np.float32),
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
    timestamps = np.asarray(timestamps, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
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
        n_resampled=int(sum(segment.n_resampled for segment in segments)),
        n_interpolated=int(sum(segment.n_interpolated for segment in segments)),
        n_segments=len(segments),
        gap=report_as_dict(gaps),
    )
