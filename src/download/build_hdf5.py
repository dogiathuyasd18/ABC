
import csv
from pathlib import Path

import h5py
import numpy as np

from list_files import (
    load_session,
    trim_and_validate_timestamps,
    validate_quaternions,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data/processed/passed_sessions.csv"

HF_MOUNT = Path("/private/tmp/boxrr23")

OUTPUT_PATH = PROJECT_ROOT / "data/processed/head_pose.h5"

ERROR_PATH = PROJECT_ROOT / "data/processed/hdf5_errors.csv"

MAX_SESSIONS = None

QUATERNION_ORDER = "x,y,z,w"

HDF5_COMPRESSION = "gzip"
HDF5_COMPRESSION_LEVEL = 4


# ============================================================
# RESOLVE TAR ARCHIVE
# ============================================================

def resolve_archive(user_id):

    if not user_id:
        raise ValueError("Missing user_id")

    archive_path = (
        HF_MOUNT
        / "users"
        / user_id[0]
        / f"{user_id}.tar"
    )

    if not archive_path.exists():
        raise FileNotFoundError(
            f"TAR archive not found: {archive_path}"
        )

    return archive_path


# ============================================================
# LOAD PASS SESSIONS FROM CSV
# ============================================================

def load_passed_sessions():

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH}"
        )

    sessions = []

    seen_keys = set()

    with open(
        CSV_PATH,
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        required_columns = {
            "user_id",
            "session_id",
            "member_name",
            "status",
        }

        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        missing_columns = (
            required_columns - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                f"Missing CSV columns: {missing_columns}"
            )

        for row in reader:

            if row["status"].strip().upper() != "PASS":
                continue

            user_id = row["user_id"].strip()
            session_id = row["session_id"].strip()
            member_name = row["member_name"].strip()

            if not user_id or not session_id or not member_name:
                continue

            key = (user_id, session_id)

            if key in seen_keys:
                continue

            seen_keys.add(key)

            sessions.append({
                "user_id": user_id,
                "session_id": session_id,
                "member_name": member_name,
            })

    return sessions


# ============================================================
# CHECK EXISTING HDF5 SESSION
# ============================================================

def is_session_complete(h5_file, user_id, session_id):

    session_path = (
        f"users/{user_id}/sessions/{session_id}"
    )

    if session_path not in h5_file:
        return False

    group = h5_file[session_path]

    if "timestamps" not in group:
        return False

    if "quaternions" not in group:
        return False

    timestamps = group["timestamps"]
    quaternions = group["quaternions"]

    if timestamps.ndim != 1:
        return False

    if quaternions.ndim != 2:
        return False

    if quaternions.shape[1] != 4:
        return False

    if timestamps.shape[0] != quaternions.shape[0]:
        return False

    if timestamps.shape[0] < 2:
        return False

    return True


# ============================================================
# SAVE SESSION TO HDF5
# ============================================================

def save_session(
    h5_file,
    user_id,
    session_id,
    member_name,
    archive_path,
    timestamps,
    quaternions,
    sampling_rate,
):

    user_group = h5_file.require_group(
        f"users/{user_id}"
    )

    sessions_group = user_group.require_group(
        "sessions"
    )

    if session_id in sessions_group:
        del sessions_group[session_id]

    session_group = sessions_group.create_group(
        session_id
    )

    session_group.create_dataset(
        "timestamps",
        data=timestamps.astype(np.float64),
        dtype=np.float64,
        compression=HDF5_COMPRESSION,
        compression_opts=HDF5_COMPRESSION_LEVEL,
        shuffle=True,
    )

    session_group.create_dataset(
        "quaternions",
        data=quaternions.astype(np.float32),
        dtype=np.float32,
        compression=HDF5_COMPRESSION,
        compression_opts=HDF5_COMPRESSION_LEVEL,
        shuffle=True,
    )

    session_group.attrs["user_id"] = user_id
    session_group.attrs["session_id"] = session_id
    session_group.attrs["member_name"] = member_name
    session_group.attrs["source_archive"] = str(archive_path)

    session_group.attrs["num_samples"] = len(timestamps)
    session_group.attrs["sampling_rate"] = float(sampling_rate)

    session_group.attrs["quaternion_order"] = QUATERNION_ORDER

    session_group.attrs["trim_seconds"] = 5.0
    session_group.attrs["timestamps_unit"] = "seconds"

    session_group.attrs["sampling_rate_valid"] = True

    h5_file.flush()


# ============================================================
# PROCESS ONE SESSION
# ============================================================

def process_session(h5_file, session):

    user_id = session["user_id"]
    session_id = session["session_id"]
    member_name = session["member_name"]

    archive_path = resolve_archive(user_id)

    print("\n" + "=" * 70)
    print("User ID:", user_id)
    print("Session ID:", session_id)
    print("Archive:", archive_path)
    print("Member:", member_name)

    # --------------------------------------------------------
    # Check existing HDF5 data
    # --------------------------------------------------------

    if is_session_complete(
        h5_file,
        user_id,
        session_id,
    ):

        print("SKIPPED: Session already exists in HDF5")

        return "SKIPPED"

    # --------------------------------------------------------
    # Load original XROR data
    # --------------------------------------------------------

    timestamps, quaternions = load_session(
        archive_path,
        member_name,
    )

    print("Original samples:", len(timestamps))
    print("Original quaternion shape:", quaternions.shape)

    # --------------------------------------------------------
    # Trim and validate timestamps
    # --------------------------------------------------------

    (
        timestamps,
        quaternions,
        sampling_rate,
        sampling_rate_valid,
    ) = trim_and_validate_timestamps(
        timestamps,
        quaternions,
    )

    print("Samples after trimming:", len(timestamps))
    print(f"Sampling rate: {sampling_rate:.4f} Hz")

    if not sampling_rate_valid:
        raise ValueError(
            f"Sampling rate outside accepted range: "
            f"{sampling_rate:.4f} Hz"
        )

    # --------------------------------------------------------
    # Validate and normalize quaternions
    # --------------------------------------------------------

    quaternions = validate_quaternions(
        quaternions
    )

    if len(timestamps) != len(quaternions):
        raise ValueError(
            "Timestamp/quaternion length mismatch after validation"
        )

    if not np.all(np.isfinite(timestamps)):
        raise ValueError(
            "Non-finite timestamps after preprocessing"
        )

    if not np.all(np.isfinite(quaternions)):
        raise ValueError(
            "Non-finite quaternions after preprocessing"
        )

    quaternion_norms = np.linalg.norm(
        quaternions,
        axis=1,
    )

    if not np.allclose(
        quaternion_norms,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Quaternion normalization failed"
        )

    # --------------------------------------------------------
    # Save HDF5
    # --------------------------------------------------------

    save_session(
        h5_file=h5_file,
        user_id=user_id,
        session_id=session_id,
        member_name=member_name,
        archive_path=archive_path,
        timestamps=timestamps,
        quaternions=quaternions,
        sampling_rate=sampling_rate,
    )

    print("Saved samples:", len(timestamps))
    print("Quaternion shape:", quaternions.shape)
    print("HDF5 save: SUCCESS")

    return "SUCCESS"


# ============================================================
# LOG ERRORS
# ============================================================

def log_error(session, error):

    ERROR_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = (
        ERROR_PATH.exists()
        and ERROR_PATH.stat().st_size > 0
    )

    with open(
        ERROR_PATH,
        "a",
        newline="",
        encoding="utf-8",
    ) as error_file:

        fieldnames = [
            "user_id",
            "session_id",
            "member_name",
            "error",
        ]

        writer = csv.DictWriter(
            error_file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "user_id": session["user_id"],
            "session_id": session["session_id"],
            "member_name": session["member_name"],
            "error": str(error),
        })


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========== BUILD HDF5 ==========")

    print("Project root:", PROJECT_ROOT)
    print("CSV:", CSV_PATH)
    print("HF mount:", HF_MOUNT)
    print("Output:", OUTPUT_PATH)
    print("Max sessions:", MAX_SESSIONS)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load PASS sessions
    # --------------------------------------------------------

    sessions = load_passed_sessions()

    print("\nUsers with PASS sessions:", len({
        session["user_id"]
        for session in sessions
    }))

    print("Total PASS sessions:", len(sessions))

    # --------------------------------------------------------
    # Process sessions
    # --------------------------------------------------------

    success_count = 0
    skipped_count = 0
    error_count = 0
    attempted_count = 0

    with h5py.File(
        OUTPUT_PATH,
        "a",
    ) as h5_file:

        h5_file.attrs["dataset"] = "BOXRR-23"
        h5_file.attrs["task"] = "Head Pose Prediction"
        h5_file.attrs["quaternion_order"] = QUATERNION_ORDER

        for index, session in enumerate(
            sessions,
            start=1,
        ):

            user_id = session["user_id"]
            session_id = session["session_id"]

            if is_session_complete(
                h5_file,
                user_id,
                session_id,
            ):

                skipped_count += 1

                print(
                    f"\n[{index}/{len(sessions)}] "
                    f"User: {user_id} | Session: {session_id}"
                )

                print("SKIPPED: Already saved")

                continue

            if (
                MAX_SESSIONS is not None
                and attempted_count >= MAX_SESSIONS
            ):
                break

            attempted_count += 1

            print(
                f"\n[{index}/{len(sessions)}] "
                f"User: {user_id} | Session: {session_id}"
            )

            try:

                result = process_session(
                    h5_file,
                    session,
                )

                if result == "SUCCESS":
                    success_count += 1

                elif result == "SKIPPED":
                    skipped_count += 1

            except Exception as error:

                error_count += 1

                print("ERROR:", str(error))

                log_error(
                    session,
                    error,
                )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n========== SUMMARY ==========")

    print("Successfully saved:", success_count)
    print("Already existed:", skipped_count)
    print("Errors:", error_count)
    print("New sessions attempted:", attempted_count)

    print("HDF5:", OUTPUT_PATH)
    print("Error log:", ERROR_PATH)

    print("=============================")


if __name__ == "__main__":
    main()
