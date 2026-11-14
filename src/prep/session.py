
"""One-session preprocessing: extract -> filter -> clean -> resample."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

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


def _fail(
    filtered: FilterResult,
    reason: str,
    clean: CleanResult | None = None,
    resample: ResampleResult | None = None,
    quality_before: dict | None = None,
    quality_after: dict | None = None,
    schema: dict | None = None,
) -> PreparedSession:
    filtered.status = "FAIL"
    filtered.reject_reason = reason

    return PreparedSession(
        filter=filtered,
        clean=clean,
        resample=resample,
        quality_before=quality_before,
        quality_after=quality_after,
        schema=schema if schema is not None else {},
        trainable=False,
        n_windows=0,
    )


def _validate_resampled_segments(segments) -> list:
    if not segments:
        raise ValueError("no_valid_segments")

    validated = []

    for index, segment in enumerate(segments):
        timestamps = np.asarray(
            segment.timestamps,
            dtype=np.float64,
        )
        quaternions = np.asarray(
            segment.quaternions,
            dtype=np.float64,
        )

        if timestamps.ndim != 1:
            raise ValueError(
                f"segment_{index}:timestamps_not_1d"
            )

        if quaternions.ndim != 2 or quaternions.shape[1] != 4:
            raise ValueError(
                f"segment_{index}:invalid_quaternion_shape"
            )

        if len(timestamps) == 0:
            raise ValueError(
                f"segment_{index}:empty_segment"
            )

        if len(timestamps) != len(quaternions):
            raise ValueError(
                f"segment_{index}:timestamp_quaternion_length_mismatch"
            )

        if not np.all(np.isfinite(timestamps)):
            raise ValueError(
                f"segment_{index}:non_finite_timestamps"
            )

        if not np.all(np.isfinite(quaternions)):
            raise ValueError(
                f"segment_{index}:non_finite_quaternions"
            )

        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError(
                f"segment_{index}:timestamps_not_strictly_increasing"
            )

        norms = np.linalg.norm(quaternions, axis=1)

        if np.any(norms < 1e-8):
            raise ValueError(
                f"segment_{index}:near_zero_quaternion"
            )

        if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
            raise ValueError(
                f"segment_{index}:quaternions_not_normalized"
            )

        if int(segment.n_resampled) != len(timestamps):
            raise ValueError(
                f"segment_{index}:n_resampled_mismatch"
            )

        validated.append(segment)

    return validated


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
    schema = schema if schema is not None else {}

    try:
        filtered = filter_session(
            timestamps,
            quaternions,
            filter_cfg,
        )
    except Exception as exc:
        raise RuntimeError(
            f"filter_stage_failed:{type(exc).__name__}:{exc}"
        ) from exc

    if not filtered.passed:
        return PreparedSession(
            filter=filtered,
            clean=None,
            resample=None,
            quality_before=None,
            quality_after=None,
            schema=schema,
            trainable=False,
            n_windows=0,
        )

    try:
        cleaned = clean_session(
            filtered.timestamps_trim,
            filtered.quaternions_trim,
            clean_cfg,
        )
    except Exception as exc:
        return _fail(
            filtered,
            f"clean:{type(exc).__name__}:{exc}",
            schema=schema,
        )

    if cleaned.rejected:
        return _fail(
            filtered,
            f"clean:{cleaned.reject_reason}",
            clean=cleaned,
            schema=schema,
        )

    if len(cleaned.timestamps) < 2:
        return _fail(
            filtered,
            "clean:insufficient_samples",
            clean=cleaned,
            schema=schema,
        )

    try:
        before = quality_as_dict(
            evaluate_sequence(
                cleaned.timestamps,
                cleaned.quaternions,
                clean_cfg,
                expected_dt=1.0 / filtered.sampling_rate_hz,
            )
        )
    except Exception as exc:
        return _fail(
            filtered,
            f"quality_before:{type(exc).__name__}:{exc}",
            clean=cleaned,
            schema=schema,
        )

    try:
        resampled = resample_session(
            cleaned.timestamps,
            cleaned.quaternions,
            clean_cfg,
        )
    except Exception as exc:
        return _fail(
            filtered,
            f"resample:{type(exc).__name__}:{exc}",
            clean=cleaned,
            quality_before=before,
            schema=schema,
        )

    try:
        valid_segments = _validate_resampled_segments(
            resampled.segments
        )
    except Exception as exc:
        return _fail(
            filtered,
            f"resample_validation:{type(exc).__name__}:{exc}",
            clean=cleaned,
            resample=resampled,
            quality_before=before,
            schema=schema,
        )

    after_parts = []

    for index, segment in enumerate(valid_segments):
        try:
            quality = evaluate_sequence(
                segment.timestamps,
                segment.quaternions,
                clean_cfg,
                expected_dt=0.01,
            )
            after_parts.append(
                quality_as_dict(quality)
            )
        except Exception as exc:
            return _fail(
                filtered,
                f"quality_after_segment_{index}:{type(exc).__name__}:{exc}",
                clean=cleaned,
                resample=resampled,
                quality_before=before,
                schema=schema,
            )

    lengths = [
        len(segment.timestamps)
        for segment in valid_segments
    ]

    try:
        trainable = session_is_trainable(
            lengths,
            window_cfg,
        )
        n_windows = count_windows(
            lengths,
            window_cfg,
        )
    except Exception as exc:
        return _fail(
            filtered,
            f"window_count:{type(exc).__name__}:{exc}",
            clean=cleaned,
            resample=resampled,
            quality_before=before,
            quality_after={"segments": after_parts},
            schema=schema,
        )

    if n_windows == 0:
        trainable = False

    return PreparedSession(
        filter=filtered,
        clean=cleaned,
        resample=resampled,
        quality_before=before,
        quality_after={"segments": after_parts},
        schema=schema,
        trainable=trainable,
        n_windows=n_windows,
    )


def prepare_from_tar(
    archive_path: Path,
    member_name: str,
    **kwargs,
) -> PreparedSession:
    try:
        extracted = extract_head_pose_from_tar(
            archive_path,
            member_name,
        )
    except Exception as exc:
        raise RuntimeError(
            f"extraction_failed:{type(exc).__name__}:{exc}"
        ) from exc

    return prepare_from_arrays(
        extracted.timestamps,
        extracted.quaternions,
        schema=extracted.schema,
        **kwargs,
    )
