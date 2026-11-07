"""Schema-driven XROR HEAD pose extraction. No hardcoded column maps."""

from __future__ import annotations

import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fpzip
import numpy as np
from bson import BSON

from src.prep.config import QUATERNION_AXES_XROR


@dataclass
class HeadExtraction:
    timestamps: np.ndarray
    quaternions: np.ndarray
    schema: dict = field(default_factory=dict)


def list_xror_members(archive_path: Path) -> list[str]:
    with tarfile.open(archive_path, "r:*") as tar:
        members = [
            member.name
            for member in tar.getmembers()
            if member.isfile() and member.name.endswith(".xror")
        ]
    members.sort()
    return members


def load_xror_document(archive_path: Path, member_name: str) -> dict:
    with tarfile.open(archive_path, "r:*") as tar:
        try:
            member = tar.getmember(member_name)
        except KeyError as error:
            raise FileNotFoundError(f"Member not found: {member_name}") from error
        file_obj = tar.extractfile(member)
        if file_obj is None:
            raise ValueError(f"Cannot extract: {member_name}")
        return BSON(file_obj.read()).decode()


def _find_head_device(devices: list) -> tuple[int, dict, int]:
    """Return (device_index, device, column_offset). Column 0 is timestamp."""
    if not isinstance(devices, list) or not devices:
        raise ValueError("Missing hardware devices metadata")

    offset = 1
    head_index = None
    head_device = None
    head_offset = None

    for index, device in enumerate(devices):
        axes = device.get("axes", [])
        if not isinstance(axes, list):
            raise ValueError(f"Invalid axes on device {index}")
        joint = str(device.get("joint", "")).upper()
        name = str(device.get("name", "")).upper()
        is_head = joint == "HEAD" or "HEAD" in name and "CONTROLLER" not in name
        if is_head:
            if head_device is not None:
                raise ValueError("Multiple HEAD devices in schema")
            head_index = index
            head_device = device
            head_offset = offset
        offset += len(axes)

    if head_device is None:
        raise ValueError("HEAD device not found in schema")
    return head_index, head_device, head_offset


def _frames_to_2d(decoded) -> tuple[np.ndarray, tuple]:
    frames = np.asarray(decoded)
    original_shape = tuple(frames.shape)
    frames = np.squeeze(frames)
    if frames.ndim != 2:
        if frames.ndim > 2:
            frames = frames.reshape(-1, frames.shape[-1])
        else:
            raise ValueError(f"Invalid frames dimensions: {original_shape}")
    return frames, original_shape


def extract_head_pose(document: dict, quaternion_axes=QUATERNION_AXES_XROR) -> HeadExtraction:
    if "frames" not in document:
        raise ValueError("Missing frames in XROR document")

    info = document.get("info", {})
    hardware = info.get("hardware", {})
    devices = hardware.get("devices", [])
    head_index, head_device, head_offset = _find_head_device(devices)
    head_axes = head_device.get("axes", [])

    for axis in quaternion_axes:
        if axis not in head_axes:
            raise ValueError(f"HEAD axes {head_axes} missing quaternion axis {axis}")

    joint = str(head_device.get("joint", "")).upper()
    if joint != "HEAD":
        raise ValueError(f"Resolved device is not HEAD joint: {head_device}")

    decoded = fpzip.decompress(document["frames"], order="C")
    frames, original_shape = _frames_to_2d(decoded)

    expected_columns = 1 + sum(len(device.get("axes", [])) for device in devices)
    if frames.shape[1] != expected_columns:
        raise ValueError(
            f"Frame columns {frames.shape[1]} != schema width {expected_columns}"
        )

    quaternion_indices = [head_offset + head_axes.index(axis) for axis in quaternion_axes]
    if max(quaternion_indices) >= frames.shape[1]:
        raise ValueError("HEAD quaternion indices exceed frame columns")

    timestamps = np.asarray(frames[:, 0], dtype=np.float64)
    quaternions = np.asarray(frames[:, quaternion_indices], dtype=np.float64)

    if timestamps.ndim != 1 or quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("Unexpected timestamp/quaternion shapes after extract")
    if len(timestamps) != len(quaternions):
        raise ValueError("Timestamp/quaternion length mismatch")
    if len(timestamps) == 0:
        raise ValueError("Empty frame data")

    schema = {
        "xror_schema": document.get("$schema"),
        "user_id_in_document": (info.get("user") or {}).get("id"),
        "n_devices": len(devices),
        "devices": [
            {
                "index": index,
                "name": device.get("name"),
                "type": device.get("type"),
                "joint": device.get("joint"),
                "axes": list(device.get("axes", [])),
            }
            for index, device in enumerate(devices)
        ],
        "head_device_index": head_index,
        "head_device_name": head_device.get("name"),
        "head_device_type": head_device.get("type"),
        "head_joint": head_device.get("joint"),
        "head_axes": list(head_axes),
        "timestamp_column": 0,
        "quaternion_columns": quaternion_indices,
        "quaternion_order_source": ",".join(quaternion_axes),
        "quaternion_order_stored": "x,y,z,w",
        "frame_shape_original": list(original_shape),
        "frame_shape_2d": list(frames.shape),
        "expected_columns": expected_columns,
        "verified_head_not_controller": True,
    }
    return HeadExtraction(timestamps=timestamps, quaternions=quaternions, schema=schema)


def extract_head_pose_from_tar(archive_path: Path, member_name: str) -> HeadExtraction:
    document = load_xror_document(archive_path, member_name)
    return extract_head_pose(document)


def schema_as_jsonable(schema: dict) -> dict:
    converted = {}
    for key, value in schema.items():
        if isinstance(value, np.ndarray):
            converted[key] = value.tolist()
        else:
            converted[key] = value
    return converted
