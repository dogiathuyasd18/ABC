"""Timestamp inspection and deterministic cleaning."""

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


def _as_timestamp_array(timestamps: np.ndarray) -> np.ndarray:
    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"timestamps must be 1D, got shape {values.shape}")
    return values


def _as_quaternion_array(quaternions: np.ndarray, expected_length: int | None = None) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"quaternions must have shape (N, 4), got {values.shape}")
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(
            f"timestamp/quaternion length mismatch: {expected_length} != {len(values)}"
        )
    return values


def infer_time_unit(median_interval: float) -> str:
    if not np.isfinite(median_interval) or median_interval <= 0:
        return "unknown"
    if 0.005 <= median_interval <= 0.02:
        return "seconds"
    if 5.0 <= median_interval <= 20.0:
        return "milliseconds_suspect"
    if 5_000.0 <= median_interval <= 20_000.0:
        return "microseconds_suspect"
    if 5_000_000.0 <= median_interval <= 20_000_000.0:
        return "nanoseconds_suspect"
    return "unexpected"


def interval_stats(
    timestamps: np.ndarray,
    cfg: FilterConfig | None = None,
    clean: CleanConfig | None = None,
) -> IntervalStats:
    cfg = cfg or FilterConfig()
    clean = clean or CleanConfig()
    timestamps = _as_timestamp_array(timestamps)

    if len(timestamps) < 2 or not np.all(np.isfinite(timestamps)):
        return IntervalStats(
            n_intervals=max(0, len(timestamps) - 1),
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
        valid = (
            n_non_monotonic == 0
            and cfg.min_sampling_rate_hz <= rate <= cfg.max_sampling_rate_hz
        )

    return IntervalStats(
        n_intervals=len(diffs),
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
        suspected_units=infer_time_unit(median),
    )


def trim_startup(
    timestamps: np.ndarray,
    quaternions: np.ndarray,
    trim_seconds: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    timestamps = _as_timestamp_array(timestamps)
    quaternions = _as_quaternion_array(quaternions, len(timestamps))
    if not np.isfinite(trim_seconds) or trim_seconds < 0:
        raise ValueError("trim_seconds must be finite and non-negative")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps contain non-finite values")
    if len(timestamps) == 0 or trim_seconds == 0:
        return timestamps.copy(), quaternions.copy(), 0
    cutoff = timestamps[0] + trim_seconds
    keep = timestamps >= cutoff
    removed = int(np.count_nonzero(~keep))
    return timestamps[keep], quaternions[keep], removed


def clean_timestamps(
    timestamps: np.ndarray,
    quaternions: np.ndarray,
    clean: CleanConfig | None = None,
) -> TimestampCleanResult:
    """Drop non-finite timestamp rows and exact duplicate samples; never sort."""
    clean = clean or CleanConfig()
    timestamps = np.asarray(timestamps, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    n_input = len(timestamps) if timestamps.ndim >= 1 else 0
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
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        return TimestampCleanResult(
            timestamps=timestamps,
            quaternions=quaternions,
            n_input=n_input,
            n_kept=n_input,
            n_removed=0,
            rejected=True,
            reject_reason="quaternions_not_n_by_4",
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

    finite_time = np.isfinite(timestamps)
    removed["non_finite_time"] = int(np.count_nonzero(~finite_time))
    timestamps = timestamps[finite_time]
    quaternions = quaternions[finite_time]

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

    kept_indices: list[int] = []
    for index in range(len(timestamps)):
        if not kept_indices:
            kept_indices.append(index)
            continue

        previous = kept_indices[-1]
        delta = timestamps[index] - timestamps[previous]
        if delta > clean.timestamp_eps:
            kept_indices.append(index)
            continue

        same_quat = np.allclose(
            quaternions[index],
            quaternions[previous],
            rtol=0.0,
            atol=clean.duplicate_quat_atol,
            equal_nan=False,
        )
        if not same_quat:
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
        removed["exact_duplicate"] += 1

    keep = np.asarray(kept_indices, dtype=np.int64)
    timestamps = timestamps[keep]
    quaternions = quaternions[keep]

    if len(timestamps) < 2:
        return TimestampCleanResult(
            timestamps=timestamps,
            quaternions=quaternions,
            n_input=n_input,
            n_kept=len(timestamps),
            n_removed=n_input - len(timestamps),
            removed_by_reason=removed,
            rejected=True,
            reject_reason="too_few_samples_after_duplicate_removal",
        )

    median_interval = float(np.median(np.diff(timestamps)))
    units = infer_time_unit(median_interval)
    if units != "seconds":
        return TimestampCleanResult(
            timestamps=timestamps,
            quaternions=quaternions,
            n_input=n_input,
            n_kept=len(timestamps),
            n_removed=n_input - len(timestamps),
            removed_by_reason=removed,
            rejected=True,
            reject_reason=f"timestamp_units_{units}",
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
