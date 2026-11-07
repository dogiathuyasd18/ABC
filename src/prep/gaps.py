"""Gap analysis and segmentation.

Gaps longer than CleanConfig.max_interpolate_gap_seconds start a new
continuous segment. Training windows must not cross those cuts.
"""

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


def gap_report(timestamps: np.ndarray, cfg: CleanConfig | None = None) -> GapReport:
    cfg = cfg or CleanConfig()
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) < 2:
        return GapReport(0, 0, 0, float("nan"), float("nan"), float("nan"), float("nan"), cfg.max_interpolate_gap_seconds, cfg.gap_threshold_rationale)
    diffs = np.diff(timestamps)
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return GapReport(0, 0, 0, float("nan"), float("nan"), float("nan"), float("nan"), cfg.max_interpolate_gap_seconds, cfg.gap_threshold_rationale)
    n_split = int(np.sum(positive > cfg.max_interpolate_gap_seconds))
    n_short = int(np.sum((positive > cfg.dropout_interval_seconds) & (positive <= cfg.max_interpolate_gap_seconds)))
    return GapReport(
        n_gaps=int(len(positive)),
        n_split=n_split,
        n_short=n_short,
        median_gap=float(np.median(positive)),
        p95_gap=float(np.percentile(positive, 95)),
        p99_gap=float(np.percentile(positive, 99)),
        max_gap=float(np.max(positive)),
        threshold_seconds=cfg.max_interpolate_gap_seconds,
        rationale=cfg.gap_threshold_rationale,
    )


def split_on_gaps(
    timestamps: np.ndarray,
    quaternions: np.ndarray,
    cfg: CleanConfig | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    cfg = cfg or CleanConfig()
    timestamps = np.asarray(timestamps, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if len(timestamps) == 0:
        return []
    if len(timestamps) == 1:
        return [(timestamps, quaternions)]

    cuts = np.where(np.diff(timestamps) > cfg.max_interpolate_gap_seconds)[0] + 1
    segments = []
    for indices in np.split(np.arange(len(timestamps)), cuts):
        if len(indices) == 0:
            continue
        segments.append((timestamps[indices], quaternions[indices]))
    return segments


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
