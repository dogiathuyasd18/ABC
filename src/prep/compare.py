"""Before/after comparison against the legacy HDF5 and filter CSV."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.prep.config import PROJECT_ROOT, CleanConfig, FilterConfig, WindowConfig
from src.prep.cleaning import clean_session
from src.prep.filtering import filter_session
from src.prep.hdf5 import iter_segments
from src.prep.resampling import resample_session
from src.prep.windows import count_windows, session_is_trainable


def compare_legacy_h5(
    h5_path: Path | None = None,
    out_path: Path | None = None,
) -> dict:
    h5_path = h5_path or (PROJECT_ROOT / "data" / "processed" / "head_pose.h5")
    out_path = out_path or (PROJECT_ROOT / "reports" / "prep_before_after.json")
    filter_cfg = FilterConfig()
    clean_cfg = CleanConfig()
    window_new = WindowConfig()
    window_old_train = WindowConfig(context_samples=100, horizon_samples=20, stride=40)
    clean_old_gap = CleanConfig(max_interpolate_gap_seconds=0.5)

    n_sessions = 0
    still_pass = 0
    fail_reasons = {}
    usable_new = 0
    usable_old_gap = 0
    windows_new = 0
    windows_old_ctx = 0
    windows_old_gap = 0
    n_segments_new = 0
    n_segments_old_gap = 0
    interpolated_new = 0
    interpolated_old = 0

    for item in iter_segments(h5_path):
        n_sessions += 1
        timestamps = item["timestamps"]
        quaternions = item["quaternions"]
        # Legacy HDF5 is already trimmed. Re-run motion+rate without a second 5 s cut
        # by using trim_seconds=0 while keeping official thresholds.
        zero_trim = FilterConfig(trim_seconds=0.0)
        filtered = filter_session(timestamps, quaternions, zero_trim)
        if filtered.passed:
            still_pass += 1
        else:
            fail_reasons[filtered.reject_reason] = fail_reasons.get(filtered.reject_reason, 0) + 1

        cleaned = clean_session(timestamps, quaternions, clean_cfg)
        if cleaned.rejected:
            continue
        new = resample_session(cleaned.timestamps, cleaned.quaternions, clean_cfg)
        old = resample_session(cleaned.timestamps, cleaned.quaternions, clean_old_gap)
        new_lengths = [segment.n_resampled for segment in new.segments]
        old_lengths = [segment.n_resampled for segment in old.segments]
        n_segments_new += new.n_segments
        n_segments_old_gap += old.n_segments
        interpolated_new += new.n_interpolated
        interpolated_old += old.n_interpolated
        if session_is_trainable(new_lengths, window_new):
            usable_new += 1
        if session_is_trainable(old_lengths, window_new):
            usable_old_gap += 1
        windows_new += count_windows(new_lengths, window_new)
        windows_old_ctx += count_windows(new_lengths, window_old_train)
        windows_old_gap += count_windows(old_lengths, window_new)

    payload = {
        "legacy_h5": str(h5_path),
        "n_legacy_sessions": n_sessions,
        "filter_still_pass_without_second_trim": still_pass,
        "filter_fail_reasons_on_legacy_rows": fail_reasons,
        "note": (
            "Legacy head_pose.h5 is already trimmed and rate-filtered. "
            "Session filtering does not use a 420-sample rule. "
            "Training usability is counted after 100 Hz resampling."
        ),
        "gap_threshold_new_s": clean_cfg.max_interpolate_gap_seconds,
        "gap_threshold_old_s": clean_old_gap.max_interpolate_gap_seconds,
        "n_segments_new_gap": n_segments_new,
        "n_segments_old_0p5s_gap": n_segments_old_gap,
        "n_interpolated_new": interpolated_new,
        "n_interpolated_old_0p5s": interpolated_old,
        "sessions_with_400_20_window_new_gap": usable_new,
        "sessions_with_400_20_window_old_gap": usable_old_gap,
        "windows_400_20_stride20_new_gap": windows_new,
        "windows_100_20_stride40_new_gap": windows_old_ctx,
        "windows_400_20_stride20_old_0p5s_gap": windows_old_gap,
        "filter_config": {
            "min_rate": filter_cfg.min_sampling_rate_hz,
            "max_rate": filter_cfg.max_sampling_rate_hz,
            "min_sweep": filter_cfg.min_sweep_degrees,
            "min_velocity": filter_cfg.min_velocity_deg_s,
            "sweep_comparison": filter_cfg.sweep_comparison,
            "spec_version": filter_cfg.spec_version,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    md = out_path.with_suffix(".md")
    md.write_text(
        "\n".join(
            [
                "# Preprocessing before / after",
                "",
                f"- legacy sessions: {n_sessions}",
                f"- still PASS on rate/sweep/velocity (no second trim): {still_pass}",
                f"- extra FAIL reasons: `{fail_reasons}`",
                f"- segments with {clean_cfg.max_interpolate_gap_seconds}s gap cut: {n_segments_new}",
                f"- segments with 0.5s gap cut: {n_segments_old_gap}",
                f"- sessions with a 400+20 window (new gap): {usable_new}",
                f"- sessions with a 400+20 window (0.5s gap): {usable_old_gap}",
                f"- windows 400+20 stride 20 (new gap): {windows_new}",
                f"- windows 100+20 stride 40 (new gap, old train default): {windows_old_ctx}",
                "",
                "Session filtering never applied a 420-sample cutoff. That check is training usability only.",
                "",
            ]
        )
    )
    return payload
