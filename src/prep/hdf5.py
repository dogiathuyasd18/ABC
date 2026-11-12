
"""HDF5 at user/session/segment sequence level. No window copies."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import h5py
import numpy as np

from src.prep.config import (
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
    SPEC_VERSION,
    WindowConfig,
    config_manifest,
)
from src.prep.resampling import ResampledSegment
from src.prep.windows import count_windows, session_is_trainable


CHUNK_SAMPLES = 4096
COMPRESSION = "gzip"
COMPRESSION_LEVEL = 4


def _checksum(timestamps: np.ndarray, quaternions: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(
        np.ascontiguousarray(timestamps, dtype=np.float64).tobytes()
    )
    hasher.update(
        np.ascontiguousarray(quaternions, dtype=np.float32).tobytes()
    )
    return hasher.hexdigest()


def _write_array(group, name, data, dtype, chunks=None):
    payload = np.asarray(data, dtype=dtype)

    if payload.ndim == 0 or payload.shape[0] == 0:
        raise ValueError(f"Cannot write empty array: {name}")

    if payload.ndim == 1:
        chunk = (min(CHUNK_SAMPLES, payload.shape[0]),)
    else:
        chunk = (
            min(CHUNK_SAMPLES, payload.shape[0]),
        ) + payload.shape[1:]

    group.create_dataset(
        name,
        data=payload,
        dtype=dtype,
        chunks=chunk,
        compression=COMPRESSION,
        compression_opts=COMPRESSION_LEVEL,
        shuffle=True,
    )


def init_file(
    path: Path,
    resampled: bool,
    filter_cfg=None,
    clean_cfg=None,
    window_cfg=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "a") as h5_file:
        h5_file.attrs["dataset"] = "BOXRR-23"
        h5_file.attrs["task"] = "Head Pose Prediction"
        h5_file.attrs["schema_version"] = SCHEMA_VERSION
        h5_file.attrs["preprocessing_version"] = PREPROCESSING_VERSION
        h5_file.attrs["spec_version"] = SPEC_VERSION
        h5_file.attrs["quaternion_order"] = "x,y,z,w"
        h5_file.attrs["resampled"] = bool(resampled)
        h5_file.attrs["target_sample_rate_hz"] = (
            100.0 if resampled else 0.0
        )
        h5_file.attrs["rate_is_native"] = not bool(resampled)
        h5_file.attrs["manifest_json"] = json.dumps(
            config_manifest(filter_cfg, clean_cfg, window_cfg)
        )
        h5_file.require_group("users")
        h5_file.flush()


def _validate_session_group(group) -> bool:
    if not bool(group.attrs.get("write_complete", False)):
        return False

    if "segments" not in group:
        return False

    segments = group["segments"]
    expected = int(group.attrs.get("n_segments", -1))

    if expected <= 0 or len(segments) != expected:
        return False

    try:
        for segment_id in segments:
            segment = segments[segment_id]

            if "timestamps" not in segment:
                return False

            if "quaternions" not in segment:
                return False

            timestamps = segment["timestamps"][:]
            quaternions = segment["quaternions"][:]

            if timestamps.ndim != 1:
                return False

            if quaternions.shape != (len(timestamps), 4):
                return False

            if len(timestamps) == 0:
                return False

            if not np.all(np.isfinite(timestamps)):
                return False

            if not np.all(np.isfinite(quaternions)):
                return False

            if np.any(np.diff(timestamps) <= 0):
                return False

            norms = np.linalg.norm(quaternions, axis=1)

            if not np.allclose(norms, 1.0, atol=1e-3):
                return False

            stored_checksum = segment.attrs.get(
                "checksum_sha256", ""
            )

            if isinstance(stored_checksum, bytes):
                stored_checksum = stored_checksum.decode("utf-8")

            if not stored_checksum:
                return False

            actual_checksum = _checksum(
                timestamps,
                quaternions,
            )

            if stored_checksum != actual_checksum:
                return False

            stored_length = int(
                segment.attrs.get("n_resampled", -1)
            )

            if stored_length != len(timestamps):
                return False

        return True

    except (KeyError, OSError, ValueError, TypeError):
        return False


def session_complete(
    h5_file,
    user_id: str,
    session_id: str,
) -> bool:
    path = f"users/{user_id}/sessions/{session_id}"

    if path not in h5_file:
        return False

    return _validate_session_group(h5_file[path])


def _verify_segment(
    group,
    timestamps: np.ndarray,
    quaternions: np.ndarray,
) -> None:
    read_t = group["timestamps"][:]
    read_q = group["quaternions"][:]

    expected_t = np.asarray(timestamps, dtype=np.float64)
    expected_q = np.asarray(quaternions, dtype=np.float32)

    if read_t.shape != expected_t.shape:
        raise ValueError("HDF5 timestamp shape mismatch")

    if read_q.shape != expected_q.shape:
        raise ValueError("HDF5 quaternion shape mismatch")

    if not np.array_equal(read_t, expected_t):
        raise ValueError("HDF5 timestamp read-back mismatch")

    if not np.array_equal(read_q, expected_q):
        raise ValueError("HDF5 quaternion read-back mismatch")

    stored = group.attrs.get("checksum_sha256", "")

    if isinstance(stored, bytes):
        stored = stored.decode("utf-8")

    actual = _checksum(read_t, read_q)

    if not stored or stored != actual:
        raise ValueError("HDF5 checksum mismatch")


def _recover_session(
    h5_file,
    user_id: str,
    session_id: str,
) -> None:
    final_path = (
        f"users/{user_id}/sessions/{session_id}"
    )
    transaction_root = (
        f"users/{user_id}/_transactions"
    )
    backup_path = (
        f"{transaction_root}/{session_id}.backup"
    )

    if transaction_root not in h5_file:
        return

    transactions = h5_file[transaction_root]

    if final_path in h5_file:
        if _validate_session_group(h5_file[final_path]):
            if backup_path in h5_file:
                del h5_file[backup_path]
        elif backup_path in h5_file:
            del h5_file[final_path]
            h5_file.move(backup_path, final_path)

    elif backup_path in h5_file:
        h5_file.move(backup_path, final_path)

    stale_prefix = f"{session_id}."

    for name in list(transactions.keys()):
        if name.startswith(stale_prefix) and name.endswith(".tmp"):
            del transactions[name]

    h5_file.flush()


def save_session(
    path: Path,
    user_id: str,
    session_id: str,
    member_name: str,
    source_archive: str,
    segments: list[ResampledSegment],
    attrs: dict,
) -> None:
    if not segments:
        raise ValueError(
            f"Cannot save session without segments: "
            f"{user_id}/{session_id}"
        )

    for index, segment in enumerate(segments):
        timestamps = np.asarray(
            segment.timestamps,
            dtype=np.float64,
        )
        quaternions = np.asarray(
            segment.quaternions,
            dtype=np.float32,
        )

        if timestamps.ndim != 1:
            raise ValueError(
                f"Invalid timestamp shape in segment {index}"
            )

        if quaternions.shape != (len(timestamps), 4):
            raise ValueError(
                f"Invalid quaternion shape in segment {index}"
            )

        if len(timestamps) == 0:
            raise ValueError(
                f"Empty segment {index}"
            )

        if not np.all(np.isfinite(timestamps)):
            raise ValueError(
                f"Non-finite timestamps in segment {index}"
            )

        if not np.all(np.isfinite(quaternions)):
            raise ValueError(
                f"Non-finite quaternions in segment {index}"
            )

        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(
                f"Non-monotonic timestamps in segment {index}"
            )

        norms = np.linalg.norm(quaternions, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                f"Non-unit quaternions in segment {index}"
            )

    with h5py.File(path, "a") as h5_file:
        user_group = h5_file.require_group(
            f"users/{user_id}"
        )
        user_group.require_group("sessions")
        transactions = user_group.require_group(
            "_transactions"
        )

        _recover_session(
            h5_file,
            user_id,
            session_id,
        )

        final_path = (
            f"users/{user_id}/sessions/{session_id}"
        )
        backup_path = (
            f"users/{user_id}/_transactions/"
            f"{session_id}.backup"
        )

        if session_complete(
            h5_file,
            user_id,
            session_id,
        ):
            return

        if final_path in h5_file:
            if backup_path in h5_file:
                del h5_file[backup_path]

            h5_file.move(final_path, backup_path)
            h5_file.flush()

        temp_name = (
            f"{session_id}.{uuid.uuid4().hex}.tmp"
        )
        temp_path = (
            f"users/{user_id}/_transactions/{temp_name}"
        )

        try:
            session = transactions.create_group(temp_name)

            session.attrs["user_id"] = user_id
            session.attrs["session_id"] = session_id
            session.attrs["member_name"] = member_name
            session.attrs["source_archive"] = source_archive
            session.attrs["schema_version"] = SCHEMA_VERSION
            session.attrs["preprocessing_version"] = (
                PREPROCESSING_VERSION
            )
            session.attrs["spec_version"] = SPEC_VERSION
            session.attrs["quaternion_order"] = "x,y,z,w"
            session.attrs["n_segments"] = len(segments)

            for key, value in attrs.items():
                if value is None:
                    continue

                if isinstance(value, (dict, list, tuple)):
                    session.attrs[key] = json.dumps(value)

                elif isinstance(
                    value,
                    (
                        bool,
                        int,
                        float,
                        str,
                        np.integer,
                        np.floating,
                    ),
                ):
                    session.attrs[key] = value

            segment_root = session.create_group("segments")

            for index, segment in enumerate(segments):
                name = f"{index:04d}"

                timestamps = np.asarray(
                    segment.timestamps,
                    dtype=np.float64,
                )
                quaternions = np.asarray(
                    segment.quaternions,
                    dtype=np.float32,
                )

                group = segment_root.create_group(name)

                _write_array(
                    group,
                    "timestamps",
                    timestamps,
                    np.float64,
                )
                _write_array(
                    group,
                    "quaternions",
                    quaternions,
                    np.float32,
                )

                group.attrs["segment_id"] = name
                group.attrs["n_source"] = segment.n_source
                group.attrs["n_resampled"] = len(timestamps)
                group.attrs["n_interpolated"] = (
                    segment.n_interpolated
                )
                group.attrs["duration_s"] = segment.duration_s
                group.attrs["checksum_sha256"] = _checksum(
                    timestamps,
                    quaternions,
                )

                _verify_segment(
                    group,
                    timestamps,
                    quaternions,
                )

            lengths = [
                len(segment.timestamps)
                for segment in segments
            ]
            window_cfg = WindowConfig()

            session.attrs["trainable"] = (
                session_is_trainable(lengths, window_cfg)
            )
            session.attrs["n_windows_default"] = (
                count_windows(lengths, window_cfg)
            )

            session.attrs["write_complete"] = True
            h5_file.flush()

            if not _validate_session_group(session):
                raise RuntimeError(
                    f"Staging session verification failed: "
                    f"{user_id}/{session_id}"
                )

            h5_file.move(temp_path, final_path)
            h5_file.flush()

            if not session_complete(
                h5_file,
                user_id,
                session_id,
            ):
                raise RuntimeError(
                    f"Final session verification failed: "
                    f"{user_id}/{session_id}"
                )

            if backup_path in h5_file:
                del h5_file[backup_path]

            h5_file.flush()

        except Exception:
            if temp_path in h5_file:
                del h5_file[temp_path]

            if backup_path in h5_file:
                if final_path in h5_file:
                    del h5_file[final_path]

                h5_file.move(backup_path, final_path)

            h5_file.flush()
            raise


def iter_segments(h5_path: Path, user_ids=None):
    with h5py.File(h5_path, "r") as h5_file:
        users = h5_file["users"]
        chosen = (
            list(user_ids)
            if user_ids is not None
            else list(users.keys())
        )

        for user_id in chosen:
            if user_id not in users:
                continue

            if "sessions" not in users[user_id]:
                continue

            sessions = users[user_id]["sessions"]

            for session_id in sessions:
                session = sessions[session_id]

                if "segments" in session:
                    for segment_id in session["segments"]:
                        group = session["segments"][segment_id]

                        yield {
                            "user_id": user_id,
                            "session_id": session_id,
                            "segment_id": str(segment_id),
                            "timestamps": group["timestamps"][:],
                            "quaternions": group["quaternions"][:],
                            "session_attrs": {
                                key: session.attrs[key]
                                for key in session.attrs
                            },
                        }

                elif "timestamps" in session:
                    yield {
                        "user_id": user_id,
                        "session_id": session_id,
                        "segment_id": "legacy",
                        "timestamps": session["timestamps"][:],
                        "quaternions": session["quaternions"][:],
                        "session_attrs": {
                            key: session.attrs[key]
                            for key in session.attrs
                        },
                    }


def list_users(h5_path: Path) -> list[str]:
    with h5py.File(h5_path, "r") as h5_file:
        if "users" not in h5_file:
            return []

        return sorted(h5_file["users"].keys())


def session_summaries(h5_path: Path) -> list[dict]:
    rows = []

    with h5py.File(h5_path, "r") as h5_file:
        if "users" not in h5_file:
            return rows

        for user_id in h5_file["users"]:
            if "sessions" not in h5_file["users"][user_id]:
                continue

            for session_id in h5_file["users"][user_id]["sessions"]:
                session = (
                    h5_file["users"][user_id]["sessions"][session_id]
                )

                if "segments" in session:
                    lengths = [
                        int(
                            session["segments"][name][
                                "quaternions"
                            ].shape[0]
                        )
                        for name in session["segments"]
                    ]

                    n_interp = sum(
                        int(
                            session["segments"][name].attrs.get(
                                "n_interpolated", 0
                            )
                        )
                        for name in session["segments"]
                    )
                else:
                    lengths = [
                        int(session["quaternions"].shape[0])
                    ]
                    n_interp = 0

                rows.append(
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "n_segments": len(lengths),
                        "segment_lengths": lengths,
                        "n_resampled": int(sum(lengths)),
                        "n_interpolated": n_interp,
                        "trainable": bool(
                            session.attrs.get(
                                "trainable",
                                session_is_trainable(lengths),
                            )
                        ),
                        "n_windows_default": int(
                            session.attrs.get(
                                "n_windows_default",
                                count_windows(lengths),
                            )
                        ),
                        "sampling_rate": float(
                            session.attrs.get(
                                "sampling_rate_hz",
                                session.attrs.get(
                                    "sampling_rate",
                                    float("nan"),
                                ),
                            )
                        ),
                    }
                )

    return rows
