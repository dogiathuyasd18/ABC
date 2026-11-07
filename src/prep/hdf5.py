"""HDF5 at user/session/segment sequence level. No window copies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from src.prep.config import (
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
    SPEC_VERSION,
    CleanConfig,
    FilterConfig,
    PathConfig,
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
    hasher.update(np.ascontiguousarray(timestamps, dtype=np.float64).tobytes())
    hasher.update(np.ascontiguousarray(quaternions, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def _write_array(group, name, data, dtype, chunks):
    if name in group:
        del group[name]
    payload = np.asarray(data, dtype=dtype)
    if payload.ndim == 1:
        chunk = (min(CHUNK_SAMPLES, max(1, payload.shape[0])),)
    else:
        chunk = (min(CHUNK_SAMPLES, max(1, payload.shape[0])),) + payload.shape[1:]
    group.create_dataset(
        name,
        data=payload,
        dtype=dtype,
        chunks=chunk,
        compression=COMPRESSION,
        compression_opts=COMPRESSION_LEVEL,
        shuffle=True,
    )


def init_file(path: Path, resampled: bool, filter_cfg=None, clean_cfg=None, window_cfg=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as h5_file:
        h5_file.attrs["dataset"] = "BOXRR-23"
        h5_file.attrs["task"] = "Head Pose Prediction"
        h5_file.attrs["schema_version"] = SCHEMA_VERSION
        h5_file.attrs["preprocessing_version"] = PREPROCESSING_VERSION
        h5_file.attrs["spec_version"] = SPEC_VERSION
        h5_file.attrs["quaternion_order"] = "x,y,z,w"
        h5_file.attrs["resampled"] = bool(resampled)
        h5_file.attrs["target_sample_rate_hz"] = 100.0 if resampled else 0.0
        h5_file.attrs["rate_is_native"] = not bool(resampled)
        h5_file.attrs["manifest_json"] = json.dumps(config_manifest(filter_cfg, clean_cfg, window_cfg))
        h5_file.require_group("users")
        h5_file.flush()


def session_complete(h5_file, user_id: str, session_id: str) -> bool:
    path = f"users/{user_id}/sessions/{session_id}"
    if path not in h5_file:
        return False
    group = h5_file[path]
    if bool(group.attrs.get("write_complete", False)):
        return True
    if "segments" not in group:
        return False
    for segment_id in group["segments"]:
        segment = group["segments"][segment_id]
        if "timestamps" not in segment or "quaternions" not in segment:
            return False
        if segment["timestamps"].shape[0] != segment["quaternions"].shape[0]:
            return False
        if segment["quaternions"].ndim != 2 or segment["quaternions"].shape[1] != 4:
            return False
    return len(group["segments"]) > 0


def _verify_segment(group, timestamps, quaternions) -> None:
    read_t = group["timestamps"][:]
    read_q = group["quaternions"][:]
    if read_t.shape != timestamps.shape or read_q.shape != np.asarray(quaternions).shape:
        raise ValueError("HDF5 read-back shape mismatch")
    if not np.allclose(read_t, timestamps):
        raise ValueError("HDF5 timestamp checksum mismatch")
    if not np.allclose(read_q, np.asarray(quaternions, dtype=np.float32), atol=1e-6):
        raise ValueError("HDF5 quaternion checksum mismatch")
    stored = group.attrs.get("checksum_sha256", "")
    actual = _checksum(read_t, read_q)
    if stored and stored != actual:
        raise ValueError("HDF5 checksum attribute mismatch")


def save_session(
    path: Path,
    user_id: str,
    session_id: str,
    member_name: str,
    source_archive: str,
    segments: list[ResampledSegment],
    attrs: dict,
) -> None:
    with h5py.File(path, "a") as h5_file:
        user_group = h5_file.require_group(f"users/{user_id}")
        sessions = user_group.require_group("sessions")
        if session_id in sessions:
            del sessions[session_id]
        session = sessions.create_group(session_id)
        session.attrs["user_id"] = user_id
        session.attrs["session_id"] = session_id
        session.attrs["member_name"] = member_name
        session.attrs["source_archive"] = source_archive
        session.attrs["schema_version"] = SCHEMA_VERSION
        session.attrs["preprocessing_version"] = PREPROCESSING_VERSION
        session.attrs["spec_version"] = SPEC_VERSION
        session.attrs["quaternion_order"] = "x,y,z,w"
        session.attrs["n_segments"] = len(segments)
        for key, value in attrs.items():
            if value is None:
                continue
            if isinstance(value, (dict, list, tuple)):
                session.attrs[key] = json.dumps(value)
            elif isinstance(value, (bool, int, float, str, np.integer, np.floating)):
                session.attrs[key] = value
        segment_root = session.create_group("segments")
        for index, segment in enumerate(segments):
            name = f"{index:04d}"
            group = segment_root.create_group(name)
            _write_array(group, "timestamps", segment.timestamps, np.float64, CHUNK_SAMPLES)
            _write_array(group, "quaternions", segment.quaternions, np.float32, CHUNK_SAMPLES)
            group.attrs["segment_id"] = name
            group.attrs["n_source"] = segment.n_source
            group.attrs["n_resampled"] = segment.n_resampled
            group.attrs["n_interpolated"] = segment.n_interpolated
            group.attrs["duration_s"] = segment.duration_s
            group.attrs["checksum_sha256"] = _checksum(segment.timestamps, segment.quaternions)
            _verify_segment(group, segment.timestamps, segment.quaternions)
        lengths = [segment.n_resampled for segment in segments]
        window_cfg = WindowConfig()
        session.attrs["trainable"] = session_is_trainable(lengths, window_cfg)
        session.attrs["n_windows_default"] = count_windows(lengths, window_cfg)
        session.attrs["write_complete"] = True
        h5_file.flush()


def iter_segments(h5_path: Path, user_ids=None):
    with h5py.File(h5_path, "r") as h5_file:
        users = h5_file["users"]
        chosen = list(user_ids) if user_ids is not None else list(users.keys())
        for user_id in chosen:
            if user_id not in users:
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
                            "session_attrs": {key: session.attrs[key] for key in session.attrs},
                        }
                elif "timestamps" in session:
                    yield {
                        "user_id": user_id,
                        "session_id": session_id,
                        "segment_id": "legacy",
                        "timestamps": session["timestamps"][:],
                        "quaternions": session["quaternions"][:],
                        "session_attrs": {key: session.attrs[key] for key in session.attrs},
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
            for session_id in h5_file["users"][user_id]["sessions"]:
                session = h5_file["users"][user_id]["sessions"][session_id]
                if "segments" in session:
                    lengths = [int(session["segments"][name]["quaternions"].shape[0]) for name in session["segments"]]
                    n_interp = sum(int(session["segments"][name].attrs.get("n_interpolated", 0)) for name in session["segments"])
                else:
                    lengths = [int(session["quaternions"].shape[0])]
                    n_interp = 0
                rows.append(
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "n_segments": len(lengths),
                        "segment_lengths": lengths,
                        "n_resampled": int(sum(lengths)),
                        "n_interpolated": n_interp,
                        "trainable": bool(session.attrs.get("trainable", session_is_trainable(lengths))),
                        "n_windows_default": int(session.attrs.get("n_windows_default", count_windows(lengths))),
                        "sampling_rate": float(session.attrs.get("sampling_rate_hz", session.attrs.get("sampling_rate", float("nan")))),
                    }
                )
    return rows
