
"""Selective BOXRR-23 download with PASS allowlist, checkpoint, retry, and resume."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from pathlib import Path

import h5py

from src.prep.config import DownloadConfig, PathConfig
from src.prep.extraction import list_xror_members
from src.prep.hdf5 import init_file, save_session, session_complete
from src.prep.session import prepare_from_tar
from src.prep.store_log import append_row

LOGGER = logging.getLogger("prep.download")

DEFAULT_PASS_CSV = Path("data/processed/passed_sessions.csv")


def _load_state(path: Path) -> dict:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {}

    state.setdefault("next_index", 0)
    state.setdefault("completed_archives", [])
    state.setdefault("failed_archives", [])
    state.setdefault("processed_members", {})
    state.setdefault("archive_results", {})

    return state


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def _retry(action, retries: int, sleep_s: float, label: str):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            return action()
        except Exception as error:
            last_error = error
            LOGGER.warning(
                "%s attempt %s/%s failed: %s",
                label,
                attempt,
                retries,
                error,
            )
            if attempt < retries:
                time.sleep(sleep_s * attempt)

    raise last_error



def _normalize_archive_path(value: str, raw_dir: Path) -> str:
    path = Path(value)

    if path.is_absolute():
        try:
            path = path.resolve().relative_to(raw_dir.resolve())
        except ValueError as error:
            raise ValueError(
                f"Archive path is outside raw directory: {value}"
            ) from error
    else:
        parts = path.parts

        if len(parts) >= 2 and parts[:2] == ("data", "raw"):
            path = Path(*parts[2:])
        elif parts and parts[0] == raw_dir.name:
            path = Path(*parts[1:])

    return path.as_posix()



def load_pass_sessions(
    csv_path: Path,
    raw_dir: Path,
) -> dict[str, set[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"PASS CSV not found: {csv_path}"
        )

    targets: dict[str, set[str]] = {}

    with csv_path.open(
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "archive",
            "member_name",
            "status",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"PASS CSV missing columns: {sorted(missing)}"
            )

        for row in reader:
            if row["status"].strip().upper() != "PASS":
                continue

            archive = _normalize_archive_path(
                row["archive"].strip(),
                raw_dir,
            )

            member_name = row["member_name"].strip()

            if not archive or not member_name:
                raise ValueError(
                    f"Invalid PASS CSV row: {row}"
                )

            targets.setdefault(archive, set()).add(member_name)

    if not targets:
        raise ValueError(
            f"No PASS sessions found in {csv_path}"
        )

    return targets


def list_remote_archives(
    cfg: DownloadConfig,
    pass_csv: Path = DEFAULT_PASS_CSV,
    raw_dir: Path = Path("data/raw"),
) -> list[str]:
    from huggingface_hub import HfApi

    targets = load_pass_sessions(pass_csv, raw_dir)

    api = HfApi()
    files = api.list_repo_files(
        repo_id=cfg.repo_id,
        repo_type=cfg.repo_type,
    )

    remote_archives = {
        path
        for path in files
        if path.startswith("users/") and path.endswith(".tar")
    }

    target_archives = set(targets)

    missing = target_archives - remote_archives

    if missing:
        examples = sorted(missing)[:10]
        raise FileNotFoundError(
            f"{len(missing)} PASS archives are missing from Hugging Face. "
            f"Examples: {examples}"
        )

    return sorted(target_archives)


def download_archive(
    remote_path: str,
    local_dir: Path,
    cfg: DownloadConfig,
) -> Path:
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

    return _retry(
        _run,
        cfg.max_retries,
        cfg.retry_sleep_seconds,
        remote_path,
    )


def persist_session(
    h5_path: Path,
    archive: Path,
    member_name: str,
    prepared,
) -> str:
    user_id = archive.stem
    session_id = Path(member_name).stem

    with h5py.File(h5_path, "a") as h5_file:
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
        "removed_by_reason": (
            prepared.clean.removed_by_reason
            if prepared.clean else {}
        ),
        "n_interpolated": prepared.resample.n_interpolated,
        "n_resampled": prepared.resample.n_resampled,
        "gap": prepared.resample.gap,
        "gap_threshold_s": prepared.resample.gap.get(
            "gap_threshold_s"
        ),
        "gap_threshold_rationale": prepared.resample.gap.get(
            "gap_threshold_rationale"
        ),
        "extraction_schema": prepared.schema,
        "quality_before": prepared.quality_before,
        "quality_after": prepared.quality_after,
        "trainable": prepared.trainable,
        "n_windows_default": prepared.n_windows,
        "filter_status": prepared.filter.status,
        "native_sampling_rate_hz": prepared.filter.sampling_rate_hz,
        "resampled": True,
        "head_device_index": prepared.schema.get(
            "head_device_index"
        ),
        "head_axes": prepared.schema.get("head_axes"),
        "quaternion_columns": prepared.schema.get(
            "quaternion_columns"
        ),
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

    with h5py.File(h5_path, "r") as h5_file:
        if not session_complete(h5_file, user_id, session_id):
            raise RuntimeError(
                f"HDF5 verification failed: {user_id}/{session_id}"
            )

    return "SAVED"


def process_local_archive(
    archive: Path,
    h5_path: Path,
    session_log: Path,
    reject_log: Path,
    target_members: set[str] | None = None,
    delete_tar: bool = True,
) -> dict:
    from src.prep.filtering import filter_row

    summary = {
        "archive": str(archive),
        "target": 0,
        "saved": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "completed": False,
    }

    try:
        archive_members = set(list_xror_members(archive))
    except Exception:
        LOGGER.exception("Cannot list archive %s", archive)
        summary["errors"] += 1
        return summary

    if target_members is None:
        target_members = archive_members
    else:
        target_members = set(target_members)

    summary["target"] = len(target_members)

    if not target_members:
        LOGGER.error("No target XROR members found in archive %s", archive)
        summary["errors"] += 1
        return summary

    missing_members = target_members - archive_members

    if missing_members:
        LOGGER.error(
            "Archive %s is missing %s target members",
            archive,
            len(missing_members),
        )
        summary["errors"] += len(missing_members)

    persisted_ok = not missing_members

    for member_name in sorted(target_members & archive_members):
        user_id = archive.stem
        session_id = Path(member_name).stem

        try:
            prepared = prepare_from_tar(archive, member_name)

            row = filter_row(
                prepared.filter,
                user_id,
                session_id,
                member_name,
                str(archive),
            )

            row["trainable"] = prepared.trainable
            row["n_windows"] = prepared.n_windows

            if not prepared.filter.passed:
                row["persist_status"] = "NOT_SAVED"
                append_row(reject_log, row)
                summary["failed"] += 1
                persisted_ok = False
                continue

            status = persist_session(
                h5_path,
                archive,
                member_name,
                prepared,
            )

            row["persist_status"] = status
            append_row(session_log, row)

            if status == "SAVED":
                summary["saved"] += 1
            elif status == "SKIPPED":
                summary["skipped"] += 1
            else:
                summary["failed"] += 1
                persisted_ok = False
                append_row(reject_log, row)

        except Exception as error:
            LOGGER.exception(
                "Session %s/%s failed",
                archive,
                member_name,
            )

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
                    "persist_status": "ERROR",
                    "reject_reason": f"exception:{error}",
                },
            )

    if persisted_ok:
        verified = True

        with h5py.File(h5_path, "r") as h5_file:
            for member_name in target_members:
                user_id = archive.stem
                session_id = Path(member_name).stem

                if not session_complete(
                    h5_file,
                    user_id,
                    session_id,
                ):
                    verified = False
                    LOGGER.error(
                        "Session not verified in HDF5: %s/%s",
                        user_id,
                        session_id,
                    )

        persisted_ok = verified

    summary["completed"] = persisted_ok

    if delete_tar and persisted_ok and archive.exists():
        archive.unlink()
        LOGGER.info(
            "Deleted TAR after verifying all PASS sessions: %s",
            archive,
        )
    elif delete_tar and not persisted_ok:
        LOGGER.warning(
            "Keeping TAR because processing was incomplete: %s",
            archive,
        )

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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pass_csv = getattr(
        paths,
        "passed_sessions_csv",
        DEFAULT_PASS_CSV,
    )

    init_file(paths.resampled_h5, resampled=True)

    targets = load_pass_sessions(pass_csv, paths.raw_dir)

    archives = list_remote_archives(
        download_cfg,
        pass_csv=pass_csv,
        raw_dir=paths.raw_dir,
    )

    state = _load_state(paths.download_state)

    completed = set(state["completed_archives"])
    failed = set(state["failed_archives"])

    archives = archives[start:]

    if max_archives is not None:
        archives = archives[:max_archives]

    LOGGER.info(
        "PASS sessions=%s target archives=%s",
        sum(len(members) for members in targets.values()),
        len(targets),
    )

    totals = {
        "saved": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "downloaded": 0,
        "completed_archives": 0,
        "failed_archives": 0,
    }

    for remote in archives:
        if remote in completed:
            LOGGER.info("Archive already completed: %s", remote)
            continue

        if (
            shutil.disk_usage(paths.raw_dir).free
            < download_cfg.min_free_gb * 1024**3
        ):
            LOGGER.warning(
                "Stopping: free disk below %.1f GB",
                download_cfg.min_free_gb,
            )
            break

        try:
            local = download_archive(
                remote,
                paths.raw_dir,
                download_cfg,
            )

            totals["downloaded"] += 1

            summary = process_local_archive(
                archive=local,
                h5_path=paths.resampled_h5,
                session_log=paths.session_log,
                reject_log=paths.reject_log,
                target_members=targets[remote],
                delete_tar=delete_tar,
            )

            for key in ("saved", "failed", "skipped", "errors"):
                totals[key] += summary[key]

            state["archive_results"][remote] = summary

            if summary["completed"]:
                if remote not in state["completed_archives"]:
                    state["completed_archives"].append(remote)

                state["failed_archives"] = [
                    item
                    for item in state["failed_archives"]
                    if item != remote
                ]

                totals["completed_archives"] += 1

            else:
                if remote not in state["failed_archives"]:
                    state["failed_archives"].append(remote)

                totals["failed_archives"] += 1

            state["processed_members"][remote] = sorted(
                targets[remote]
            )

        except Exception as error:
            LOGGER.exception("Archive failed: %s", remote)

            state["archive_results"][remote] = {
                "archive": remote,
                "completed": False,
                "error": str(error),
            }

            if remote not in state["failed_archives"]:
                state["failed_archives"].append(remote)

            totals["errors"] += 1
            totals["failed_archives"] += 1

        state["next_index"] = start + archives.index(remote) + 1
        _save_state(paths.download_state, state)

    LOGGER.info("Download totals: %s", totals)

    return totals
