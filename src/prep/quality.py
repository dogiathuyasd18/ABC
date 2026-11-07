"""Continuity checks on native and resampled sequences."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.prep.config import CleanConfig, TARGET_SAMPLE_INTERVAL
from src.prep.quat_np import consecutive_geodesic_degrees, norms, unit_norm_ok


@dataclass
class SequenceQuality:
    n_samples: int
    all_finite: bool
    unit_norm_ok: bool
    max_norm_error: float
    mean_dt: float
    median_dt: float
    max_abs_dt_error: float
    mean_step_deg: float
    max_step_deg: float
    n_discontinuities: int
    mean_velocity_deg_s: float
    max_velocity_deg_s: float


def evaluate_sequence(
    timestamps: np.ndarray,
    quaternions: np.ndarray,
    cfg: CleanConfig | None = None,
    expected_dt: float | None = None,
) -> SequenceQuality:
    cfg = cfg or CleanConfig()
    timestamps = np.asarray(timestamps, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    n_samples = len(timestamps)
    finite = bool(np.isfinite(timestamps).all() and np.isfinite(quaternions).all())
    lengths = norms(quaternions) if n_samples else np.zeros(0)
    max_norm_error = float(np.max(np.abs(lengths - 1.0))) if n_samples else float("nan")
    ok_norm = unit_norm_ok(quaternions, cfg.unit_norm_atol) if n_samples else False

    if n_samples < 2:
        return SequenceQuality(
            n_samples=n_samples,
            all_finite=finite,
            unit_norm_ok=ok_norm,
            max_norm_error=max_norm_error,
            mean_dt=float("nan"),
            median_dt=float("nan"),
            max_abs_dt_error=float("nan"),
            mean_step_deg=float("nan"),
            max_step_deg=float("nan"),
            n_discontinuities=0,
            mean_velocity_deg_s=float("nan"),
            max_velocity_deg_s=float("nan"),
        )

    diffs = np.diff(timestamps)
    steps = consecutive_geodesic_degrees(quaternions)
    dt = np.clip(diffs, cfg.timestamp_eps, None)
    velocity = steps / dt
    expect = TARGET_SAMPLE_INTERVAL if expected_dt is None else expected_dt
    return SequenceQuality(
        n_samples=n_samples,
        all_finite=finite,
        unit_norm_ok=ok_norm,
        max_norm_error=max_norm_error,
        mean_dt=float(np.mean(diffs)),
        median_dt=float(np.median(diffs)),
        max_abs_dt_error=float(np.max(np.abs(diffs - expect))),
        mean_step_deg=float(np.mean(steps)),
        max_step_deg=float(np.max(steps)),
        n_discontinuities=int(np.sum(steps > cfg.discontinuity_degrees)),
        mean_velocity_deg_s=float(np.mean(velocity)),
        max_velocity_deg_s=float(np.max(velocity)),
    )


def quality_as_dict(quality: SequenceQuality) -> dict:
    return asdict(quality)
