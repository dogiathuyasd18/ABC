"""CLI for the separated preparation stages."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.prep.compare import compare_legacy_h5
from src.prep.config import PathConfig
from src.prep.download import process_local_archive, run_download
from src.prep.hdf5 import init_file
from src.prep.report import write_quality_report


LOGGER = logging.getLogger("prep.pipeline")


def cmd_filter_local(args) -> None:
    paths = PathConfig()
    init_file(paths.resampled_h5, resampled=True)
    archives = sorted(Path(args.raw_dir).rglob("*.tar"))
    LOGGER.info("Local archives: %s", len(archives))
    for archive in archives:
        process_local_archive(
            archive,
            paths.resampled_h5,
            paths.session_log,
            paths.reject_log,
            delete_tar=args.delete_tar,
        )


def cmd_download(args) -> None:
    run_download(start=args.start, max_archives=args.max_archives, delete_tar=args.delete_tar)


def cmd_report(args) -> None:
    paths = PathConfig()
    h5_path = Path(args.h5) if args.h5 else (
        paths.resampled_h5 if paths.resampled_h5.exists() else paths.legacy_h5
    )
    pass_log = paths.session_log if paths.session_log.exists() else paths.processed_dir / "passed_sessions.csv"
    write_quality_report(paths.report_dir, pass_log, paths.reject_log, h5_path)


def cmd_migrate(args) -> None:
    from src.prep.cleaning import clean_session
    from src.prep.config import CleanConfig
    from src.prep.hdf5 import init_file, save_session
    from src.prep.resampling import resample_session
    from src.prep.quality import evaluate_sequence, quality_as_dict
    import h5py

    src = Path(args.h5) if args.h5 else PathConfig().legacy_h5
    dst = Path(args.out) if args.out else PathConfig().resampled_h5
    init_file(dst, resampled=True)
    clean_cfg = CleanConfig()
    n_ok = 0
    with h5py.File(src, "r") as h5_file:
        for user_id in h5_file["users"]:
            for session_id in h5_file["users"][user_id]["sessions"]:
                group = h5_file["users"][user_id]["sessions"][session_id]
                timestamps = group["timestamps"][:]
                quaternions = group["quaternions"][:]
                cleaned = clean_session(timestamps, quaternions, clean_cfg)
                if cleaned.rejected:
                    LOGGER.warning("legacy %s/%s clean reject: %s", user_id, session_id, cleaned.reject_reason)
                    continue
                resampled = resample_session(cleaned.timestamps, cleaned.quaternions, clean_cfg)
                attrs = {
                    "sampling_rate_hz": float(group.attrs.get("sampling_rate", float("nan"))),
                    "n_raw": int(group.attrs.get("num_samples", len(timestamps))),
                    "n_after_trim": len(timestamps),
                    "n_clean": cleaned.n_kept,
                    "n_removed": cleaned.n_removed,
                    "removed_by_reason": cleaned.removed_by_reason,
                    "n_interpolated": resampled.n_interpolated,
                    "n_resampled": resampled.n_resampled,
                    "gap": resampled.gap,
                    "gap_threshold_s": resampled.gap.get("gap_threshold_s"),
                    "gap_threshold_rationale": resampled.gap.get("gap_threshold_rationale"),
                    "quality_before": quality_as_dict(evaluate_sequence(cleaned.timestamps, cleaned.quaternions, clean_cfg, expected_dt=1.0 / max(float(group.attrs.get("sampling_rate", 100.0)), 1e-6))),
                    "migrated_from_legacy": True,
                    "resampled": True,
                    "member_name": str(group.attrs.get("member_name", "")),
                    "source_archive": str(group.attrs.get("source_archive", "")),
                }
                save_session(
                    dst,
                    user_id=str(user_id),
                    session_id=str(session_id),
                    member_name=str(group.attrs.get("member_name", "")),
                    source_archive=str(group.attrs.get("source_archive", "")),
                    segments=resampled.segments,
                    attrs=attrs,
                )
                n_ok += 1
    LOGGER.info("Migrated %s sessions -> %s", n_ok, dst)


def cmd_compare(args) -> None:
    payload = compare_legacy_h5(Path(args.h5) if args.h5 else None)
    print(payload)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="9th ABC data preparation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    filt = sub.add_parser("filter-local", help="Filter TAR files already on disk")
    filt.add_argument("--raw-dir", default=str(PathConfig().raw_dir))
    filt.add_argument("--delete-tar", action="store_true")
    filt.set_defaults(func=cmd_filter_local)

    down = sub.add_parser("download", help="Download BOXRR-23 in resumable batches")
    down.add_argument("--start", type=int, default=0)
    down.add_argument("--max-archives", type=int, default=None)
    down.add_argument("--delete-tar", action="store_true")
    down.set_defaults(func=cmd_download)

    report = sub.add_parser("report", help="Write the data quality report")
    report.add_argument("--h5", default="")
    report.set_defaults(func=cmd_report)

    compare = sub.add_parser("compare", help="Compare new rules against legacy HDF5")
    compare.add_argument("--h5", default="")
    compare.set_defaults(func=cmd_compare)

    migrate = sub.add_parser("migrate-legacy", help="Rebuild v2 HDF5 from legacy head_pose.h5")
    migrate.add_argument("--h5", default="")
    migrate.add_argument("--out", default="")
    migrate.set_defaults(func=cmd_migrate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
