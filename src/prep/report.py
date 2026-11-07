"""Data quality report: session filtering vs training usability."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.prep.config import (
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
    SPEC_VERSION,
    WindowConfig,
    config_manifest,
)
from src.prep.gaps import gap_report, report_as_dict
from src.prep.hdf5 import iter_segments, list_users, session_summaries
from src.prep.windows import count_windows, session_is_trainable, window_starts


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _floats(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _hist(path: Path, values: list[float], title: str, xlabel: str) -> None:
    if not values:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(values, bins=min(40, max(10, int(np.sqrt(len(values))))) )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_filter_report(pass_rows: list[dict], reject_rows: list[dict]) -> dict:
    all_rows = pass_rows + reject_rows
    reasons = Counter(row.get("reject_reason") or "PASS" for row in all_rows)
    users = {row.get("user_id") for row in all_rows if row.get("user_id")}
    return {
        "n_users_inspected": len(users),
        "n_sessions_inspected": len(all_rows),
        "n_pass": sum(1 for row in all_rows if str(row.get("status", "")).upper() == "PASS"),
        "n_fail": sum(1 for row in all_rows if str(row.get("status", "")).upper() != "PASS"),
        "reject_reasons": dict(reasons),
        "sampling_rate_hz": _floats(all_rows, "sampling_rate"),
        "session_duration_proxy_samples": _floats(all_rows, "frames_after_trim"),
        "session_sweep": _floats(all_rows, "session_sweep"),
        "session_velocity": _floats(all_rows, "session_velocity"),
        "duplicate_timestamps": _floats(all_rows, "interval_n_duplicate_timestamps"),
        "non_monotonic_timestamps": _floats(all_rows, "interval_n_non_monotonic_timestamps"),
    }


def build_usability_report(h5_path: Path, window_cfg: WindowConfig | None = None) -> dict:
    window_cfg = window_cfg or WindowConfig()
    summaries = session_summaries(h5_path)
    users = list_users(h5_path)
    windows_by_user = defaultdict(int)
    trainable = 0
    not_trainable = 0
    total_windows = 0
    n_segments = 0
    n_resampled = 0
    n_interpolated = 0
    for row in summaries:
        n_segments += row["n_segments"]
        n_resampled += row["n_resampled"]
        n_interpolated += row["n_interpolated"]
        n_win = count_windows(row["segment_lengths"], window_cfg)
        windows_by_user[row["user_id"]] += n_win
        total_windows += n_win
        if session_is_trainable(row["segment_lengths"], window_cfg):
            trainable += 1
        else:
            not_trainable += 1
    return {
        "n_users": len(users),
        "n_sessions": len(summaries),
        "n_segments": n_segments,
        "n_resampled_samples": n_resampled,
        "n_interpolated_samples": n_interpolated,
        "n_sessions_with_window": trainable,
        "n_sessions_without_window": not_trainable,
        "n_usable_windows": total_windows,
        "windows_per_user": dict(windows_by_user),
        "window_config": {
            "context": window_cfg.context_samples,
            "horizon": window_cfg.horizon_samples,
            "stride": window_cfg.stride,
        },
    }


def write_quality_report(
    report_dir: Path,
    pass_log: Path,
    reject_log: Path,
    h5_path: Path,
    seed: int = 0,
) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    pass_rows = [row for row in _read_csv(pass_log) if str(row.get("status", "")).upper() == "PASS"]
    # session_filter.csv may contain only PASS; reject log holds FAILs.
    reject_rows = _read_csv(reject_log)
    extra_pass = [row for row in _read_csv(pass_log)]
    filter_report = build_filter_report(extra_pass, reject_rows)
    usability = build_usability_report(h5_path) if h5_path.exists() else {}

    native_gaps = []
    if h5_path.exists():
        for item in iter_segments(h5_path):
            native_gaps.extend(np.diff(item["timestamps"]).tolist())

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "spec_version": SPEC_VERSION,
        "seed": seed,
        "manifest": config_manifest(),
        "session_filtering": {
            **{key: value for key, value in filter_report.items() if not isinstance(value, list)},
            "distributions": {
                "sampling_rate_hz": _summary(filter_report["sampling_rate_hz"]),
                "frames_after_trim": _summary(filter_report["session_duration_proxy_samples"]),
                "session_sweep": _summary(filter_report["session_sweep"]),
                "session_velocity": _summary(filter_report["session_velocity"]),
            },
        },
        "training_usability": usability,
        "inter_sample_gaps_resampled": _summary(native_gaps),
    }
    json_path = report_dir / "data_quality_report.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    md_path = report_dir / "data_quality_report.md"
    md_path.write_text(_markdown(payload))

    _hist(report_dir / "hist_sampling_rate.png", filter_report["sampling_rate_hz"], "Sampling rate", "Hz")
    _hist(report_dir / "hist_sweep.png", filter_report["session_sweep"], "Session sweep", "degrees")
    _hist(report_dir / "hist_velocity.png", filter_report["session_velocity"], "Session velocity", "deg/s")
    _hist(report_dir / "hist_windows_per_user.png", list(usability.get("windows_per_user", {}).values()), "Windows per user", "windows")
    if native_gaps:
        _hist(report_dir / "hist_resampled_dt.png", native_gaps[:200000], "Resampled intervals", "seconds")
    return payload


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _markdown(payload: dict) -> str:
    filt = payload["session_filtering"]
    use = payload.get("training_usability") or {}
    lines = [
        "# Data quality report",
        "",
        f"- spec: `{payload['spec_version']}`",
        f"- preprocessing: `{payload['preprocessing_version']}`",
        f"- schema: `{payload['schema_version']}`",
        "",
        "## Session filtering",
        "",
        f"- users inspected: {filt.get('n_users_inspected')}",
        f"- sessions inspected: {filt.get('n_sessions_inspected')}",
        f"- PASS: {filt.get('n_pass')}",
        f"- FAIL: {filt.get('n_fail')}",
        f"- reject reasons: `{filt.get('reject_reasons')}`",
        "",
        "## Training usability (after clean / segment / 100 Hz)",
        "",
        f"- sessions with ≥1 window of 400+20: {use.get('n_sessions_with_window')}",
        f"- sessions with no window: {use.get('n_sessions_without_window')}",
        f"- usable windows: {use.get('n_usable_windows')}",
        f"- segments: {use.get('n_segments')}",
        f"- resampled samples: {use.get('n_resampled_samples')}",
        f"- interpolated samples: {use.get('n_interpolated_samples')}",
        "",
    ]
    return "\n".join(lines)
