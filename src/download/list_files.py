
import tarfile

import fpzip
import numpy as np

from bson import BSON


# ============================================================
# CONFIGURATION
# ============================================================

TRIM_SECONDS = 5.0

MIN_SAMPLING_RATE = 85.0
MAX_SAMPLING_RATE = 110.0

MIN_SESSION_SAMPLES = 20

QUATERNION_ORDER = ["i", "j", "k", "1"]

DEBUG = True


# ============================================================
# DEBUG FUNCTIONS
# ============================================================

def debug_timestamp_columns(frames):

    print("\n========== FRAME DEBUG ==========")
    print("Frames shape:", frames.shape)
    print("Frames dtype:", frames.dtype)

    print("\nFirst 5 rows:")
    print(frames[:5])

    print("\n========== COLUMN ANALYSIS ==========")

    for col in range(frames.shape[1]):

        values = frames[:, col]
        differences = np.diff(values)

        if len(differences) == 0:
            continue

        print(
            f"Column {col:02d} | "
            f"min={np.min(values):.6f} | "
            f"max={np.max(values):.6f} | "
            f"median_diff={np.median(differences):.8f} | "
            f"positive={np.sum(differences > 0)} | "
            f"zero={np.sum(differences == 0)} | "
            f"negative={np.sum(differences < 0)}"
        )

    print("====================================\n")


# ============================================================
# LOAD XROR SESSION
# ============================================================

def load_session(archive_path, member_name):

    # --------------------------------------------------------
    # Read BSON
    # --------------------------------------------------------

    with tarfile.open(archive_path, "r:*") as tar:

        try:
            member = tar.getmember(member_name)
        except KeyError:
            raise FileNotFoundError(
                f"Member not found: {member_name}"
            )

        file_obj = tar.extractfile(member)

        if file_obj is None:
            raise ValueError(
                f"Cannot extract: {member_name}"
            )

        document = BSON(file_obj.read()).decode()

    # --------------------------------------------------------
    # Validate metadata
    # --------------------------------------------------------

    if "frames" not in document:
        raise ValueError("Missing frames in XROR document")

    info = document.get("info", {})
    hardware = info.get("hardware", {})
    devices = hardware.get("devices", [])

    if not isinstance(devices, list) or not devices:
        raise ValueError("Missing hardware devices metadata")

    # --------------------------------------------------------
    # Find HEAD device
    # --------------------------------------------------------

    head_index = None
    head_device = None

    for index, device in enumerate(devices):

        if device.get("joint") == "HEAD":
            head_index = index
            head_device = device
            break

    if head_device is None:
        raise ValueError("HEAD device not found")

    axes = head_device.get("axes", [])

    if not isinstance(axes, list):
        raise ValueError("Invalid HEAD axes metadata")

    for axis in QUATERNION_ORDER:

        if axis not in axes:
            raise ValueError(
                f"Missing quaternion axis: {axis}"
            )

    # --------------------------------------------------------
    # Calculate HEAD offset
    # --------------------------------------------------------

    device_offset = 1

    for device in devices[:head_index]:

        device_axes = device.get("axes", [])

        if not isinstance(device_axes, list):
            raise ValueError("Invalid device axes metadata")

        device_offset += len(device_axes)

    quaternion_indices = [
        device_offset + axes.index(axis)
        for axis in QUATERNION_ORDER
    ]

    # --------------------------------------------------------
    # Decompress FPZIP
    # --------------------------------------------------------

    frames = fpzip.decompress(
        document["frames"],
        order="C",
    )

    frames = np.asarray(frames)

    original_shape = frames.shape

    # Remove singleton dimensions
    frames = np.squeeze(frames)

    if frames.ndim != 2:
        raise ValueError(
            f"Invalid frames dimensions: "
            f"original={original_shape}, "
            f"after squeeze={frames.shape}"
        )

    # --------------------------------------------------------
    # Validate frame layout
    # --------------------------------------------------------

    expected_columns = 1 + sum(
        len(device.get("axes", []))
        for device in devices
    )

    if frames.shape[1] != expected_columns:
        raise ValueError(
            f"Unexpected frame layout: "
            f"shape={frames.shape}, "
            f"expected_columns={expected_columns}"
        )

    if max(quaternion_indices) >= frames.shape[1]:
        raise ValueError(
            f"Quaternion indices exceed frame columns: "
            f"{quaternion_indices}"
        )

    if DEBUG:

        print("\n========== XROR METADATA ==========")
        print("Schema:", document.get("$schema"))
        print("User:", info.get("user", {}).get("id"))
        print("Devices:")

        for index, device in enumerate(devices):
            print(
                index,
                device.get("name"),
                device.get("type"),
                device.get("joint"),
                device.get("axes"),
            )

        print("\nOriginal frames shape:", original_shape)
        print("Processed frames shape:", frames.shape)
        print("HEAD device index:", head_index)
        print("HEAD device offset:", device_offset)
        print("HEAD quaternion indices:", quaternion_indices)
        print("===================================\n")

        debug_timestamp_columns(frames)

    # --------------------------------------------------------
    # Extract candidate timestamp and HEAD quaternion
    # --------------------------------------------------------

    timestamps = np.asarray(
        frames[:, 0],
        dtype=np.float64,
    )

    quaternions = np.asarray(
        frames[:, quaternion_indices],
        dtype=np.float64,
    )

    if len(timestamps) == 0:
        raise ValueError("Empty frame data")

    if not np.isfinite(timestamps).all():
        raise ValueError("Non-finite timestamps")

    if not np.isfinite(quaternions).all():
        raise ValueError("Non-finite quaternion values")

    if DEBUG:

        print("Candidate timestamp first 10:")
        print(timestamps[:10])

        print("\nCandidate quaternion first 5:")
        print(quaternions[:5])

        print("\nCandidate quaternion norms:")
        print(
            np.linalg.norm(
                quaternions[:5],
                axis=1,
            )
        )

    return timestamps, quaternions


# ============================================================
# TRIM AND VALIDATE TIMESTAMPS
# ============================================================


def trim_and_validate_timestamps(
    timestamps,
    quaternions,
):

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    quaternions = np.asarray(
        quaternions,
        dtype=np.float64,
    )

    if timestamps.ndim != 1:
        raise ValueError("Timestamps must be a 1D array")

    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(
            f"Invalid quaternion shape: {quaternions.shape}"
        )

    if len(timestamps) != len(quaternions):
        raise ValueError("Timestamp/quaternion length mismatch")

    if len(timestamps) < 2:
        raise ValueError("Not enough timestamps")

    if not np.isfinite(timestamps).all():
        raise ValueError("Non-finite timestamps")

    if not np.isfinite(quaternions).all():
        raise ValueError("Non-finite quaternion values")

    original_count = len(timestamps)

    # Trim first 5 seconds before timestamp validation.
    recording_start = timestamps[0]
    trim_timestamp = recording_start + TRIM_SECONDS

    trim_mask = timestamps >= trim_timestamp

    timestamps = timestamps[trim_mask]
    quaternions = quaternions[trim_mask]

    if len(timestamps) < MIN_SESSION_SAMPLES:
        raise ValueError(
            "Fewer than minimum session samples after trimming"
        )

    # Validate timestamps after trimming.
    time_diffs = np.diff(timestamps)

    if len(time_diffs) == 0:
        raise ValueError(
            "Not enough timestamps to estimate sampling rate"
        )

    if np.any(time_diffs <= 0):
        raise ValueError(
            "Duplicate or non-monotonic timestamps after trimming"
        )

    median_interval = np.median(time_diffs)

    sampling_rate = 1.0 / median_interval

    sampling_rate_valid = (
        MIN_SAMPLING_RATE
        <= sampling_rate
        <= MAX_SAMPLING_RATE
    )

    if DEBUG:
        print("\n========== TIMESTAMP VALIDATION ==========")
        print("Original samples:", original_count)
        print("Samples after trimming:", len(timestamps))
        print("Trimmed samples:", original_count - len(timestamps))
        print("Duplicate timestamps after trimming:", np.sum(time_diffs == 0))
        print("Median interval:", median_interval)
        print("Sampling rate:", sampling_rate)
        print("Sampling rate valid:", sampling_rate_valid)
        print("==========================================\n")

    return (
        timestamps,
        quaternions,
        sampling_rate,
        sampling_rate_valid,
    )



# ============================================================
# VALIDATE QUATERNIONS
# ============================================================

def validate_quaternions(quaternions):

    quaternions = np.asarray(
        quaternions,
        dtype=np.float64,
    )

    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(
            f"Invalid quaternion shape: {quaternions.shape}"
        )

    if len(quaternions) == 0:
        raise ValueError("Empty quaternion array")

    if not np.isfinite(quaternions).all():
        raise ValueError(
            "Non-finite quaternion values"
        )

    # --------------------------------------------------------
    # Validate quaternion norms
    # --------------------------------------------------------

    norms = np.linalg.norm(
        quaternions,
        axis=1,
    )

    if np.any(norms < 1e-8):
        raise ValueError(
            "Near-zero quaternion detected"
        )

    # Normalize
    quaternions = quaternions / norms[:, None]

    # --------------------------------------------------------
    # Hemisphere continuity
    # --------------------------------------------------------

    for i in range(1, len(quaternions)):

        dot = np.dot(
            quaternions[i - 1],
            quaternions[i],
        )

        if dot < 0:
            quaternions[i] *= -1

    return quaternions.astype(np.float32)
