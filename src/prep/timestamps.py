"""Timestamp inspection and deterministic cleaning.

Never sort to hide order errors. Never extrapolate. Exact duplicate rows may
be dropped; the same timestamp with two different quaternions rejects the
session.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.prep.config import CleanConfig, FilterConfig


@dataclass
class IntervalStats:
    n_intervals: int
    median: float
    mean: float
    std: float
    p95: float
    maximum: float
    n_duplicate: int
    n_non_monotonic: int
    n_dropout: int
    sampling_rate_hz: float
    sampling_rate_valid: bool
    suspected_units: str


@dataclass
class TimestampCleanResult:
    timestamps: np.ndarray
    quaternions: np.ndarray
    n_input: int
    n_kept: int
    n_removed: int
    removed_by_reason: dict = field(default_factory=dict)
    rejected: bool = False
    reject_reason: str = ""


def infer_time_unit(median_interval: float) -> str:
    if not np.isfinite(median_interval) or median_interval <= 0:
        return "unknown"
    if 0.005 <= median_interval <= 0.02:
        return "seconds"
    if 5.0 <= median_interval <= 20.0:
        return "milliseconds_suspect"
    return "unexpected"


def interval_stats(timestamps: np.ndarray, cfg: FilterConfig | None = None, clean: CleanConfig | None = None) -> IntervalStats:
    cfg = cfg or FilterConfig()
    clean = clean or CleanConfig()
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) < 2:
        return IntervalStats(
            n_intervals=0,
            median=float("nan"),
            mean=float("nan"),
            std=float("nan"),
            p95=float("nan"),
            maximum=float("nan"),
            n_duplicate=0,
            n_non_monotonic=0,
            n_dropout=0,
            sampling_rate_hz=float("nan"),
            sampling_rate_valid=False,
            suspected_units="unknown",
        )

    diffs = np.diff(timestamps)
    n_duplicate = int(np.sum(np.abs(diffs) <= clean.timestamp_eps))
    n_non_monotonic = int(np.sum(diffs < -clean.timestamp_eps))
    positive = diffs[diffs > clean.timestamp_eps]
    n_dropout = int(np.sum(positive > clean.dropout_interval_seconds)) if len(positive) else 0

    if len(positive) == 0:
        median = mean = std = p95 = maximum = float("nan")
        rate = float("nan")
        valid = False
    else:
        median = float(np.median(positive))
        mean = float(np.mean(positive))
        std = float(np.std(positive))
        p95 = float(np.percentile(positive, 95))
        maximum = float(np.max(positive))
        rate = 1.0 / median
        valid = cfg.min_sampling_rate_hz <= rate <= cfg.max_sampling_rate_hz

    return IntervalStats(
        n_intervals=int(len(diffs)),
        median=median,
        mean=mean,
        std=std,
        p95=p95,
        maximum=maximum,
        n_duplicate=n_duplicate,
        n_non_monotonic=n_non_monotonic,
        n_dropout=n_dropout,
        sampling_rate_hz=float(rate),
        sampling_rate_valid=bool(valid),
        suspected_units=infer_time_unit(median if np.isfinite(median) else float("nan")),
    )


def trim_startup(timestamps: np.ndarray, quaternions: np.ndarray, trim_seconds: float) -> tuple[np.ndarray, np.ndarray, int]:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if len(timestamps) == 0:
        return timestamps, quaternions, 0
    cutoff = timestamps[0] + trim_seconds
    keep = timestamps >= cutoff
    removed = int((~keep).sum())
    return timestamps[keep], quaternions[keep], removed


def clean_timestamps(
    timestamps: np.ndarray,
    quaternions: np.ndarray,
    clean: CleanConfig | None = None,
) -> TimestampCleanResult:
    """Apply documented duplicate rules. Do not sort. Do not invent times."""
    clean = clean or CleanConfig()
    timestamps = np.asarray(timestamps, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    n_input = len(timestamps)
    removed = {"non_finite_time": 0, "exact_duplicate": 0}

    if timestamps.ndim != 1:
        return TimestampCleanResult(
            timestamps=timestamps,
            quaternions=quaternions,
            n_input=n_input,
            n_kept=n_input,
            n_removed=0,
            rejected=True,
            reject_reason="timestamps_not_1d",
        )
    if len(timestamps) != len(quaternions):
        return TimestampCleanResult(
            timestamps=timestamps,
            quaternions=quaternions,
            n_input=n_input,
            n_kept=n_input,
            n_removed=0,
            rejected=True,
            reject_reason="timestamp_quaternion_length_mismatch",
        )

    finite = np.isfinite(timestamps)
    removed["non_finite_time"] = int((~finite).sum())
    timestamps = timestamps[finite]
    quaternions = quaternions[finite]

    if len(timestamps) < 2:
        return TimestampCleanResult(
            timestamps=timestamps,
            quaternions=quaternions,
            n_input=n_input,
            n_kept=len(timestamps),
            n_removed=n_input - len(timestamps),
            removed_by_reason=removed,
            rejected=True,
            reject_reason="too_few_finite_timestamps",
        )

    diffs = np.diff(timestamps)
    if np.any(diffs < -clean.timestamp_eps):
        return TimestampCleanResult(
            timestamps=timestamps,
            quaternions=quaternions,
            n_input=n_input,
            n_kept=len(timestamps),
            n_removed=n_input - len(timestamps),
            removed_by_reason=removed,
            rejected=True,
            reject_reason="non_monotonic_timestamps",
        )

    keep = np.ones(len(timestamps), dtype=bool)
    for index in range(1, len(timestamps)):
        if timestamps[index] - timestamps[index - 1] > clean.timestamp_eps:
            continue
        same_quat = np.allclose(
            quaternions[index],
            quaternions[index - 1],
            atol=clean.duplicate_quat_atol,
        )
        if same_quat:
            keep[index] = False
            removed["exact_duplicate"] += 1
        else:
            return TimestampCleanResult(
                timestamps=timestamps,
                quaternions=quaternions,
                n_input=n_input,
                n_kept=len(timestamps),
                n_removed=n_input - len(timestamps),
                removed_by_reason=removed,
                rejected=True,
                reject_reason="duplicate_timestamp_conflicting_quaternion",
            )

    timestamps = timestamps[keep]
    quaternions = quaternions[keep]
    units = infer_time_unit(float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else float("nan"))
    if units == "milliseconds_suspect":
        return TimestampCleanResult(
            timestamps=timestamps,
            quaternions=quaternions,
            n_input=n_input,
            n_kept=len(timestamps),
            n_removed=n_input - len(timestamps),
            removed_by_reason=removed,
            rejected=True,
            reject_reason="timestamp_units_not_seconds",
        )

    return TimestampCleanResult(
        timestamps=timestamps,
        quaternions=quaternions,
        n_input=n_input,
        n_kept=len(timestamps),
        n_removed=n_input - len(timestamps),
        removed_by_reason=removed,
        rejected=False,
    )


def stats_as_dict(stats: IntervalStats) -> dict:
    return {
        "n_intervals": stats.n_intervals,
        "median_interval_s": stats.median,
        "mean_interval_s": stats.mean,
        "std_interval_s": stats.std,
        "p95_interval_s": stats.p95,
        "max_interval_s": stats.maximum,
        "n_duplicate_timestamps": stats.n_duplicate,
        "n_non_monotonic_timestamps": stats.n_non_monotonic,
        "n_dropout_intervals": stats.n_dropout,
        "sampling_rate_hz": stats.sampling_rate_hz,
        "sampling_rate_valid": stats.sampling_rate_valid,
        "timestamp_units": stats.suspected_units,
    }
