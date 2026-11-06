
import csv
import random
import tarfile
from pathlib import Path

import fpzip
import numpy as np

from bson import BSON
from scipy.spatial.transform import Rotation


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path("data/raw/users")
OUTPUT_PATH = Path("data/processed/passed_sessions.csv")

# Dynamic mode: inspect every TAR archive currently present under DATA_ROOT.
# Change the thresholds below only when you intentionally want different criteria.

TRIM_SECONDS = 5.0
WINDOW_SIZE = 5.0

MIN_SAMPLING_RATE = 85.0
MAX_SAMPLING_RATE = 110.0

MIN_SWEEP = 17.0
MIN_VELOCITY = 15.0

MIN_VALID_SAMPLES = 10
MIN_SESSION_SAMPLES = 20

QUATERNION_ORDER = ["i", "j", "k", "1"]
EULER_ORDER = "xyz"

AXIS_NAMES = ["X", "Y", "Z"]


# ============================================================
# 1. LOAD ONE XROR SESSION
# ============================================================

def load_session(archive_path, member_name):

    with tarfile.open(archive_path, "r") as tar:

        file_obj = tar.extractfile(member_name)

        if file_obj is None:
            raise RuntimeError(
                f"Cannot read XROR member: {member_name}"
            )

        raw = file_obj.read()

    document = BSON(raw).decode()

    info = document["info"]
    devices = info.get("hardware", {}).get("devices", [])

    if not devices:
        raise RuntimeError("No devices found")

    # Find HEAD device and its column offset.
    current_offset = 1
    head_device = None
    head_offset = None

    for device in devices:

        axes = device.get("axes", [])

        if device.get("joint") == "HEAD":

            if head_device is not None:
                raise RuntimeError("Multiple HEAD devices found")

            head_device = device
            head_offset = current_offset

        current_offset += len(axes)

    if head_device is None:
        raise RuntimeError("HEAD device not found")

    head_axes = head_device.get("axes", [])

    if not all(axis in head_axes for axis in QUATERNION_ORDER):
        raise RuntimeError(
            f"Unexpected HEAD axes: {head_axes}"
        )

    # Decompress frames.
    decoded = fpzip.decompress(
        document["frames"],
        order="C"
    )

    if decoded.size == 0:
        raise RuntimeError("Decoded frame array is empty")

    frames = decoded.reshape(
        -1,
        decoded.shape[-1]
    )

    expected_columns = 1 + sum(
        len(device.get("axes", []))
        for device in devices
    )

    if frames.shape[1] != expected_columns:
        raise RuntimeError(
            "Frame column count does not match device schema"
        )

    timestamps = frames[:, 0].astype(np.float64)

    head_data = frames[
        :,
        head_offset:head_offset + len(head_axes)
    ]

    axis_indices = [
        head_axes.index(axis)
        for axis in QUATERNION_ORDER
    ]

    quaternions = head_data[
        :,
        axis_indices
    ].astype(np.float64)

    return timestamps, quaternions


# ============================================================
# 2. TRIM AND VALIDATE TIMESTAMPS
# ============================================================

def trim_and_validate_timestamps(timestamps, quaternions):

    if len(timestamps) == 0:
        raise RuntimeError("No timestamps found")

    if not np.all(np.isfinite(timestamps)):
        raise RuntimeError("Non-finite timestamps detected")

    recording_start = timestamps[0]
    trim_timestamp = recording_start + TRIM_SECONDS

    trim_mask = timestamps >= trim_timestamp

    timestamps = timestamps[trim_mask]
    quaternions = quaternions[trim_mask]

    if len(timestamps) < MIN_SESSION_SAMPLES:
        raise RuntimeError(
            "Fewer than minimum session samples after trimming"
        )

    time_diffs = np.diff(timestamps)

    if len(time_diffs) == 0:
        raise RuntimeError(
            "Not enough timestamps to estimate sampling rate"
        )

    if np.any(time_diffs <= 0):
        raise RuntimeError(
            "Duplicate or non-monotonic timestamps"
        )

    median_interval = np.median(time_diffs)
    sampling_rate = 1.0 / median_interval

    sampling_rate_valid = (
        MIN_SAMPLING_RATE
        <= sampling_rate
        <= MAX_SAMPLING_RATE
    )

    return timestamps, quaternions, sampling_rate, sampling_rate_valid


# ============================================================
# 3. VALIDATE AND NORMALIZE QUATERNIONS
# ============================================================

def validate_quaternions(quaternions):

    if not np.all(np.isfinite(quaternions)):
        raise RuntimeError("Non-finite quaternion values")

    quat_norms = np.linalg.norm(
        quaternions,
        axis=1
    )

    if np.any(quat_norms < 1e-8):
        raise RuntimeError("Near-zero quaternion detected")

    quaternions = (
        quaternions
        / quat_norms[:, np.newaxis]
    )

    # Ensure quaternion sign continuity.
    quaternions = quaternions.copy()

    for i in range(1, len(quaternions)):

        dot = np.dot(
            quaternions[i - 1],
            quaternions[i]
        )

        if dot < 0:
            quaternions[i] *= -1

    return quaternions


# ============================================================
# 4. CONVERT QUATERNIONS TO EULER ANGLES
# ============================================================

def convert_to_euler(quaternions):

    rotation = Rotation.from_quat(quaternions)

    euler_angles = rotation.as_euler(
        EULER_ORDER,
        degrees=True
    )

    if not np.all(np.isfinite(euler_angles)):
        raise RuntimeError("Non-finite Euler angles")

    return euler_angles


# ============================================================
# 5. ANALYZE WINDOWS
# ============================================================

def analyze_windows(timestamps, euler_angles):

    window_results = []

    start_time = timestamps[0]
    end_time = timestamps[-1]

    window_start = start_time

    while window_start + WINDOW_SIZE <= end_time:

        window_end = window_start + WINDOW_SIZE

        sample_mask = (
            (timestamps >= window_start)
            &
            (timestamps < window_end)
        )

        window_times = timestamps[sample_mask]
        window_angles = euler_angles[sample_mask]

        if len(window_times) < MIN_VALID_SAMPLES:
            window_start += WINDOW_SIZE
            continue

        # Step 4: Sweep = P90 - P10.
        sweep = (
            np.percentile(
                window_angles,
                90,
                axis=0
            )
            -
            np.percentile(
                window_angles,
                10,
                axis=0
            )
        )

        # Select consecutive intervals fully inside the window.
        interval_mask = (
            (timestamps[:-1] >= window_start)
            &
            (timestamps[1:] < window_end)
        )

        previous_angles = euler_angles[:-1][interval_mask]
        next_angles = euler_angles[1:][interval_mask]

        if len(previous_angles) == 0:
            window_start += WINDOW_SIZE
            continue

        # Step 5: Absolute differences.
        angle_diffs = np.abs(
            next_angles - previous_angles
        )

        # Wrap differences greater than 180 degrees.
        wrapped_diffs = np.where(
            angle_diffs > 180.0,
            360.0 - angle_diffs,
            angle_diffs
        )

        velocity = (
            np.sum(
                wrapped_diffs,
                axis=0
            )
            / WINDOW_SIZE
        )

        window_results.append({
            "sweep": sweep,
            "velocity": velocity,
            "num_samples": len(window_times),
            "num_intervals": len(previous_angles)
        })

        window_start += WINDOW_SIZE

    if not window_results:
        raise RuntimeError("No valid 5-second windows found")

    return window_results


# ============================================================
# 6. AGGREGATE SESSION METRICS
# ============================================================

def aggregate_session(window_results):

    all_sweeps = np.array([
        result["sweep"]
        for result in window_results
    ])

    all_velocities = np.array([
        result["velocity"]
        for result in window_results
    ])

    # Step 6: Median across windows, independently per axis.
    median_sweep = np.median(
        all_sweeps,
        axis=0
    )

    median_velocity = np.median(
        all_velocities,
        axis=0
    )

    session_sweep = np.max(median_sweep)
    session_velocity = np.max(median_velocity)

    return (
        median_sweep,
        median_velocity,
        session_sweep,
        session_velocity
    )


# ============================================================
# 7. INSPECT ONE SESSION
# ============================================================

def inspect_session(archive_path, member_name):

    timestamps, quaternions = load_session(
        archive_path,
        member_name
    )

    (
        timestamps,
        quaternions,
        sampling_rate,
        sampling_rate_valid
    ) = trim_and_validate_timestamps(
        timestamps,
        quaternions
    )

    quaternions = validate_quaternions(quaternions)

    euler_angles = convert_to_euler(quaternions)

    window_results = analyze_windows(
        timestamps,
        euler_angles
    )

    (
        median_sweep,
        median_velocity,
        session_sweep,
        session_velocity
    ) = aggregate_session(window_results)

    sweep_valid = session_sweep > MIN_SWEEP
    velocity_valid = session_velocity > MIN_VELOCITY

    session_valid = (
        sampling_rate_valid
        and sweep_valid
        and velocity_valid
    )

    result = {
        "archive": str(archive_path),
        "user_id": archive_path.stem,
        "session_id": Path(member_name).stem,
        "member_name": member_name,
        "frames_after_trim": len(timestamps),
        "sampling_rate": sampling_rate,
        "sampling_rate_valid": sampling_rate_valid,
        "num_valid_windows": len(window_results),

        "sweep_x": median_sweep[0],
        "sweep_y": median_sweep[1],
        "sweep_z": median_sweep[2],

        "velocity_x": median_velocity[0],
        "velocity_y": median_velocity[1],
        "velocity_z": median_velocity[2],

        "session_sweep": session_sweep,
        "session_velocity": session_velocity,

        "sweep_valid": sweep_valid,
        "velocity_valid": velocity_valid,
        "session_valid": session_valid,

        "status": "PASS" if session_valid else "FAIL",
        "reject_reason": ""
    }

    return result


# ============================================================
# 8. FIND SAMPLE ARCHIVES
# ============================================================

def find_sample_archives():
    """
    Discover every TAR archive already downloaded under DATA_ROOT.
    No fixed batch index or batch size is applied.
    """
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Data root not found: {DATA_ROOT}")

    archives = sorted(
        DATA_ROOT.rglob("*.tar"),
        key=lambda path: str(path)
    )

    if not archives:
        raise RuntimeError(f"No TAR archives found under {DATA_ROOT}")

    return archives


# ============================================================
# 9. FIND XROR MEMBER IN ARCHIVE
# ============================================================

def find_xror_members(archive_path):
    """
    Return every XROR session contained in one TAR archive.
    A TAR may contain more than one session.
    """
    with tarfile.open(archive_path, "r") as tar:
        members = [
            member.name
            for member in tar.getmembers()
            if member.isfile() and member.name.endswith(".xror")
        ]

    members.sort()

    if not members:
        raise RuntimeError("No XROR member found in archive")

    return members


# ============================================================
# 10. MAIN BATCH INSPECTION
# ============================================================



def main():
    print("\n=== DYNAMIC LOCAL DATASET FILTERING ===")

    archives = find_sample_archives()

    print("Data root:", DATA_ROOT)
    print("TAR archives found:", len(archives))
    print("Mode: inspect all downloaded archives and all XROR sessions")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "archive", "user_id", "session_id", "member_name",
        "frames_after_trim", "sampling_rate", "sampling_rate_valid",
        "num_valid_windows",
        "sweep_x", "sweep_y", "sweep_z",
        "velocity_x", "velocity_y", "velocity_z",
        "session_sweep", "session_velocity",
        "sweep_valid", "velocity_valid", "session_valid",
        "status", "reject_reason"
    ]

    # Load existing session keys to prevent duplicate CSV rows.
    existing_keys = set()

    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        with open(OUTPUT_PATH, "r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                existing_keys.add((
                    row["user_id"],
                    row["session_id"]
                ))

    print("Existing PASS sessions in CSV:", len(existing_keys))

    # Create CSV header only if the file does not exist or is empty.
    write_header = (
        not OUTPUT_PATH.exists()
        or OUTPUT_PATH.stat().st_size == 0
    )

    passed_count = 0
    skipped_count = 0
    error_count = 0

    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        if write_header:
            writer.writeheader()
            csv_file.flush()

        for archive_index, archive_path in enumerate(archives, start=1):

            print("\n" + "=" * 70)
            print(f"ARCHIVE {archive_index}/{len(archives)}")
            print("Archive:", archive_path)

            try:
                member_names = find_xror_members(archive_path)

                print("XROR sessions found:", len(member_names))

                for session_index, member_name in enumerate(member_names, start=1):

                    print(
                        f"\n  Session {session_index}/{len(member_names)}: "
                        f"{member_name}"
                    )

                    try:
                        result = inspect_session(
                            archive_path,
                            member_name
                        )

                        print(
                            f"    Sampling rate: "
                            f"{result['sampling_rate']:.4f} Hz"
                        )
                        print(
                            f"    Sweep: "
                            f"{result['session_sweep']:.4f} degrees"
                        )
                        print(
                            f"    Velocity: "
                            f"{result['session_velocity']:.4f} degrees/second"
                        )
                        print("    Status:", result["status"])

                        if result["status"] != "PASS":
                            continue

                        session_key = (
                            result["user_id"],
                            result["session_id"]
                        )

                        if session_key in existing_keys:
                            print("    Skipped: session already exists in CSV")
                            skipped_count += 1
                            continue

                        # Append this PASS session immediately.
                        writer.writerow(result)
                        csv_file.flush()

                        existing_keys.add(session_key)
                        passed_count += 1

                        print("    Saved to CSV")

                    except Exception as error:
                        error_count += 1
                        print("    ERROR:", str(error))

            except Exception as error:
                error_count += 1
                print("  ARCHIVE ERROR:", str(error))

            finally:
                # Always delete the TAR after processing.
                if archive_path.exists():
                    try:
                        archive_path.unlink()
                        print("\nDeleted TAR:", archive_path)
                    except OSError as error:
                        print("\nFailed to delete TAR:", error)

    print("\n" + "=" * 70)
    print("=== FILTER SUMMARY ===")
    print("TAR archives discovered:", len(archives))
    print("New PASS sessions:", passed_count)
    print("Duplicate sessions skipped:", skipped_count)
    print("Errors:", error_count)
    print("Total PASS sessions in CSV:", len(existing_keys))
    print("PASS CSV:", OUTPUT_PATH)
    print("=== PROCESSING COMPLETE ===")


if __name__ == "__main__":
    main()