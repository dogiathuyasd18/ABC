
"""Gap analysis and segmentation for continuous time-series samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.prep.config import CleanConfig


@dataclass
class GapReport:
    n_gaps: int
    n_split: int
    n_short: int
    median_gap: float
    p95_gap: float
    p99_gap: float
    max_gap: float
    threshold_seconds: float
    rationale: str


def _validate_timestamps(timestamps: np.ndarray) -> np.ndarray:
    values = np.asarray(timestamps, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError(
            f"timestamps must be 1D, got shape {values.shape}"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("timestamps contain non-finite values")

    if len(values) >= 2 and np.any(np.diff(values) <= 0):
        raise ValueError(
            "timestamps must be strictly increasing before gap analysis"
        )

    return values


def _validate_quaternions(
    quaternions: np.ndarray,
    expected_length: int,
) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)

    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(
            f"quaternions must have shape (N, 4), got {values.shape}"
        )

    if len(values) != expected_length:
        raise ValueError("timestamp_quaternion_length_mismatch")

    if not np.all(np.isfinite(values)):
        raise ValueError("quaternions contain non-finite values")

    return values


def _validate_thresholds(
    cfg: CleanConfig,
) -> tuple[float, float]:
    split_threshold = float(cfg.max_interpolate_gap_seconds)
    dropout_threshold = float(cfg.dropout_interval_seconds)

    if not np.isfinite(split_threshold) or split_threshold <= 0:
        raise ValueError(
            "max_interpolate_gap_seconds must be finite and positive"
        )

    if not np.isfinite(dropout_threshold) or dropout_threshold < 0:
        raise ValueError(
            "dropout_interval_seconds must be finite and non-negative"
        )

    if dropout_threshold > split_threshold:
        raise ValueError(
            "dropout_interval_seconds cannot exceed "
            "max_interpolate_gap_seconds"
        )

    return split_threshold, dropout_threshold


def gap_report(
    timestamps: np.ndarray,
    cfg: CleanConfig | None = None,
) -> GapReport:
    cfg = cfg or CleanConfig()

    timestamps = _validate_timestamps(timestamps)
    split_threshold, dropout_threshold = _validate_thresholds(cfg)

    if len(timestamps) < 2:
        return GapReport(
            n_gaps=0,
            n_split=0,
            n_short=0,
            median_gap=float("nan"),
            p95_gap=float("nan"),
            p99_gap=float("nan"),
            max_gap=float("nan"),
            threshold_seconds=split_threshold,
            rationale=cfg.gap_threshold_rationale,
        )

    intervals = np.diff(timestamps)

    n_split = int(
        np.count_nonzero(intervals > split_threshold)
    )

    n_short = int(
        np.count_nonzero(
            (intervals > dropout_threshold)
            & (intervals <= split_threshold)
        )
    )

    return GapReport(
        n_gaps=len(intervals),
        n_split=n_split,
        n_short=n_short,
        median_gap=float(np.median(intervals)),
        p95_gap=float(np.percentile(intervals, 95)),
        p99_gap=float(np.percentile(intervals, 99)),
        max_gap=float(np.max(intervals)),
        threshold_seconds=split_threshold,
        rationale=cfg.gap_threshold_rationale,
    )


def split_on_gaps(
    timestamps: np.ndarray,
    quaternions: np.ndarray,
    cfg: CleanConfig | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    cfg = cfg or CleanConfig()

    timestamps = _validate_timestamps(timestamps)
    quaternions = _validate_quaternions(
        quaternions,
        len(timestamps),
    )

    split_threshold, _ = _validate_thresholds(cfg)

    if len(timestamps) == 0:
        return []

    cuts = (
        np.flatnonzero(
            np.diff(timestamps) > split_threshold
        ) + 1
    )

    index_segments = np.split(
        np.arange(len(timestamps)),
        cuts,
    )

    return [
        (
            timestamps[index].copy(),
            quaternions[index].copy(),
        )
        for index in index_segments
        if len(index) > 0
    ]


def report_as_dict(report: GapReport) -> dict:
    return {
        "n_positive_intervals": report.n_gaps,
        "n_gaps_over_threshold": report.n_split,
        "n_short_interpolable_gaps": report.n_short,
        "median_gap_s": report.median_gap,
        "p95_gap_s": report.p95_gap,
        "p99_gap_s": report.p99_gap,
        "max_gap_s": report.max_gap,
        "gap_threshold_s": report.threshold_seconds,
        "gap_threshold_rationale": report.rationale,
    }
