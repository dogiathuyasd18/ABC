"""Batched BOXRR-23 download with checkpoint, retry, and resume.

PASS sessions are written to HDF5 and verified before a TAR is deleted.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from src.prep.config import DownloadConfig, PathConfig
from src.prep.extraction import list_xror_members
from src.prep.hdf5 import init_file, save_session, session_complete
from src.prep.resampling import ResampledSegment
from src.prep.session import prepare_from_tar
from src.prep.store_log import append_row

LOGGER = logging.getLogger("prep.download")


def _load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "next_index": 0,
        "completed_archives": [],
        "failed_archives": [],
        "processed_members": [],
    }


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def _retry(action, retries: int, sleep_s: float, label: str):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return action()
        except Exception as error:
            last_error = error
            LOGGER.warning("%s attempt %s/%s failed: %s", label, attempt, retries, error)
            time.sleep(sleep_s * attempt)
    raise last_error


def list_remote_archives(cfg: DownloadConfig) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id=cfg.repo_id, repo_type=cfg.repo_type)
    return sorted(path for path in files if path.startswith("users/") and path.endswith(".tar"))


def download_archive(remote_path: str, local_dir: Path, cfg: DownloadConfig) -> Path:
    from huggingface_hub import hf_hub_download

    def _run():
        return Path(
            hf_hub_download(
                repo_id=cfg.repo_id,
                repo_type=cfg.repo_type,
                filename=remote_path,
                local_dir=str(local_dir),
            )
        )

    return _retry(_run, cfg.max_retries, cfg.retry_sleep_seconds, remote_path)


def persist_session(h5_path: Path, archive: Path, member_name: str, prepared) -> str:
    user_id = archive.stem
    session_id = Path(member_name).stem
    with __import__("h5py").File(h5_path, "a") as h5_file:
        if session_complete(h5_file, user_id, session_id):
            return "SKIPPED"
    if not prepared.filter.passed or prepared.resample is None:
        return "FAIL"
    attrs = {
        "sampling_rate_hz": prepared.filter.sampling_rate_hz,
        "session_sweep": prepared.filter.session_sweep,
        "session_velocity": prepared.filter.session_velocity,
        "sweep_xyz": prepared.filter.sweep_xyz,
        "velocity_xyz": prepared.filter.velocity_xyz,
        "n_raw": prepared.filter.n_raw,
        "n_after_trim": prepared.filter.n_after_trim,
        "n_clean": prepared.clean.n_kept if prepared.clean else 0,
        "n_removed": prepared.clean.n_removed if prepared.clean else 0,
        "removed_by_reason": prepared.clean.removed_by_reason if prepared.clean else {},
        "n_interpolated": prepared.resample.n_interpolated,
        "n_resampled": prepared.resample.n_resampled,
        "gap": prepared.resample.gap,
        "gap_threshold_s": prepared.resample.gap.get("gap_threshold_s"),
        "gap_threshold_rationale": prepared.resample.gap.get("gap_threshold_rationale"),
        "extraction_schema": prepared.schema,
        "quality_before": prepared.quality_before,
        "quality_after": prepared.quality_after,
        "trainable": prepared.trainable,
        "n_windows_default": prepared.n_windows,
        "filter_status": prepared.filter.status,
        "native_sampling_rate_hz": prepared.filter.sampling_rate_hz,
        "resampled": True,
        "head_device_index": prepared.schema.get("head_device_index"),
        "head_axes": prepared.schema.get("head_axes"),
        "quaternion_columns": prepared.schema.get("quaternion_columns"),
    }
    save_session(
        h5_path,
        user_id=user_id,
        session_id=session_id,
        member_name=member_name,
        source_archive=str(archive),
        segments=prepared.resample.segments,
        attrs=attrs,
    )
    return "SAVED"


def process_local_archive(
    archive: Path,
    h5_path: Path,
    session_log: Path,
    reject_log: Path,
    delete_tar: bool = True,
) -> dict:
    from src.prep.filtering import filter_row

    summary = {"archive": str(archive), "saved": 0, "failed": 0, "skipped": 0, "errors": 0}
    try:
        members = list_xror_members(archive)
    except Exception as error:
        LOGGER.error("Cannot list %s: %s", archive, error)
        summary["errors"] += 1
        return summary

    persisted_ok = True
    for member_name in members:
        user_id = archive.stem
        session_id = Path(member_name).stem
        try:
            prepared = prepare_from_tar(archive, member_name)
            row = filter_row(prepared.filter, user_id, session_id, member_name, str(archive))
            row["trainable"] = prepared.trainable
            row["n_windows"] = prepared.n_windows
            if prepared.filter.passed:
                status = persist_session(h5_path, archive, member_name, prepared)
                row["persist_status"] = status
                append_row(session_log, row)
                if status == "SAVED":
                    summary["saved"] += 1
                elif status == "SKIPPED":
                    summary["skipped"] += 1
                else:
                    persisted_ok = False
                    summary["failed"] += 1
                    append_row(reject_log, row)
            else:
                row["persist_status"] = "NOT_SAVED"
                append_row(reject_log, row)
                summary["failed"] += 1
        except Exception as error:
            LOGGER.exception("Session %s/%s failed", archive, member_name)
            persisted_ok = False
            summary["errors"] += 1
            append_row(
                reject_log,
                {
                    "archive": str(archive),
                    "user_id": user_id,
                    "session_id": session_id,
                    "member_name": member_name,
                    "status": "FAIL",
                    "reject_reason": f"exception:{error}",
                },
            )

    if delete_tar and persisted_ok and archive.exists():
        archive.unlink()
        LOGGER.info("Deleted TAR after verified persist: %s", archive)
    elif delete_tar and not persisted_ok:
        LOGGER.warning("Keeping TAR because persist was incomplete: %s", archive)
    return summary


def run_download(
    start: int = 0,
    max_archives: int | None = None,
    download_cfg: DownloadConfig | None = None,
    paths: PathConfig | None = None,
    delete_tar: bool = True,
) -> dict:
    download_cfg = download_cfg or DownloadConfig()
    paths = paths or PathConfig()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_file(paths.resampled_h5, resampled=True)
    state = _load_state(paths.download_state)
    next_index = max(start, int(state.get("next_index", 0)))
    archives = list_remote_archives(download_cfg)
    end = len(archives) if max_archives is None else min(len(archives), next_index + max_archives)
    LOGGER.info("Remote archives=%s next_index=%s", len(archives), next_index)

    totals = {"saved": 0, "failed": 0, "skipped": 0, "errors": 0, "downloaded": 0}
    while next_index < end:
        if shutil.disk_usage(paths.raw_dir).free < download_cfg.min_free_gb * 1024**3:
            LOGGER.warning("Stopping: free disk below %.1f GB", download_cfg.min_free_gb)
            break
        batch = archives[next_index : next_index + download_cfg.batch_size]
        if not batch:
            break
        for remote in batch:
            local = download_archive(remote, paths.raw_dir, download_cfg)
            totals["downloaded"] += 1
            summary = process_local_archive(local, paths.resampled_h5, paths.session_log, paths.reject_log, delete_tar=delete_tar)
            for key in ("saved", "failed", "skipped", "errors"):
                totals[key] += summary[key]
            state.setdefault("completed_archives", []).append(remote)
            next_index += 1
            state["next_index"] = next_index
            _save_state(paths.download_state, state)
    LOGGER.info("Download totals: %s", totals)
    return totals
