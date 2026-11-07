"""One-session path: extract → filter → clean → resample. No download I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.prep.cleaning import CleanResult, clean_session
from src.prep.config import CleanConfig, FilterConfig, WindowConfig
from src.prep.extraction import extract_head_pose_from_tar
from src.prep.filtering import FilterResult, filter_session
from src.prep.quality import evaluate_sequence, quality_as_dict
from src.prep.resampling import ResampleResult, resample_session
from src.prep.windows import count_windows, session_is_trainable


@dataclass
class PreparedSession:
    filter: FilterResult
    clean: CleanResult | None
    resample: ResampleResult | None
    quality_before: dict | None
    quality_after: dict | None
    schema: dict
    trainable: bool
    n_windows: int


def prepare_from_arrays(
    timestamps,
    quaternions,
    filter_cfg: FilterConfig | None = None,
    clean_cfg: CleanConfig | None = None,
    window_cfg: WindowConfig | None = None,
    schema: dict | None = None,
) -> PreparedSession:
    filter_cfg = filter_cfg or FilterConfig()
    clean_cfg = clean_cfg or CleanConfig()
    window_cfg = window_cfg or WindowConfig()
    filtered = filter_session(timestamps, quaternions, filter_cfg)
    if not filtered.passed:
        return PreparedSession(filtered, None, None, None, None, schema or {}, False, 0)

    cleaned = clean_session(filtered.timestamps_trim, filtered.quaternions_trim, clean_cfg)
    if cleaned.rejected:
        fail = filtered
        fail.status = "FAIL"
        fail.reject_reason = f"clean:{cleaned.reject_reason}"
        return PreparedSession(fail, cleaned, None, None, None, schema or {}, False, 0)

    before = quality_as_dict(evaluate_sequence(cleaned.timestamps, cleaned.quaternions, clean_cfg, expected_dt=1.0 / filtered.sampling_rate_hz))
    resampled = resample_session(cleaned.timestamps, cleaned.quaternions, clean_cfg)
    after_parts = []
    for segment in resampled.segments:
        after_parts.append(quality_as_dict(evaluate_sequence(segment.timestamps, segment.quaternions, clean_cfg, expected_dt=0.01)))
    lengths = [segment.n_resampled for segment in resampled.segments]
    trainable = session_is_trainable(lengths, window_cfg)
    n_windows = count_windows(lengths, window_cfg)
    return PreparedSession(
        filter=filtered,
        clean=cleaned,
        resample=resampled,
        quality_before=before,
        quality_after={"segments": after_parts},
        schema=schema or {},
        trainable=trainable,
        n_windows=n_windows,
    )


def prepare_from_tar(archive_path: Path, member_name: str, **kwargs) -> PreparedSession:
    extracted = extract_head_pose_from_tar(archive_path, member_name)
    return prepare_from_arrays(extracted.timestamps, extracted.quaternions, schema=extracted.schema, **kwargs)
