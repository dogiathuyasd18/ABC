"""User-disjoint windows from the cleaned head-pose HDF5 file.

Sessions are resampled onto a uniform 100 Hz grid. Gaps longer than half a
second split the session instead of being invented by interpolation.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset

from src.utils.quaternions import apply_body_rotvec, geodesic_degrees, quat_inv, quat_mul, quat_to_rotvec


CONTEXT_LENGTH = 100
HORIZON = 20
SAMPLE_RATE_HZ = 100.0
MAX_GAP_SECONDS = 0.5
MIN_WINDOW = CONTEXT_LENGTH + HORIZON


def list_users(h5_path):
    with h5py.File(h5_path, "r") as h5_file:
        return sorted(h5_file["users"].keys())


def split_users(user_ids, val_fraction=0.2):
    """Hold out whole users. Windows from the same person never cross the split."""
    if not user_ids:
        raise ValueError("No users in the HDF5 file")
    val_count = max(1, int(round(len(user_ids) * val_fraction)))
    if val_count >= len(user_ids):
        val_count = 1
    return user_ids[:-val_count], user_ids[-val_count:]


def _hemisphere_align(quaternions):
    aligned = np.array(quaternions, copy=True)
    for index in range(1, len(aligned)):
        if np.dot(aligned[index - 1], aligned[index]) < 0.0:
            aligned[index] *= -1.0
    return aligned


def resample_session(timestamps, quaternions, sample_rate_hz=SAMPLE_RATE_HZ, max_gap_seconds=MAX_GAP_SECONDS):
    """SLERP each contiguous stretch onto a uniform grid. Long dropouts split the session."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    quaternions = _hemisphere_align(np.asarray(quaternions, dtype=np.float64))
    if len(timestamps) < 2:
        return []

    gap_cuts = np.where(np.diff(timestamps) > max_gap_seconds)[0] + 1
    segments = []
    step = 1.0 / sample_rate_hz

    for indices in np.split(np.arange(len(timestamps)), gap_cuts):
        if len(indices) < 2:
            continue
        segment_time = timestamps[indices]
        segment_quat = quaternions[indices]
        keep = np.ones(len(segment_time), dtype=bool)
        keep[1:] = np.diff(segment_time) > 1e-6
        segment_time = segment_time[keep]
        segment_quat = segment_quat[keep]
        if len(segment_time) < 2 or segment_time[-1] - segment_time[0] < step:
            continue

        grid = np.arange(segment_time[0], segment_time[-1], step)
        if len(grid) < MIN_WINDOW:
            continue

        resampled = Slerp(segment_time, Rotation.from_quat(segment_quat))(grid).as_quat()
        resampled = _hemisphere_align(resampled).astype(np.float32)
        segments.append(resampled)

    return segments


def load_segments(h5_path, user_ids):
    segments = []
    with h5py.File(h5_path, "r") as h5_file:
        users = h5_file["users"]
        for user_id in user_ids:
            sessions = users[user_id]["sessions"]
            for session_id in sessions:
                group = sessions[session_id]
                pieces = resample_session(group["timestamps"][:], group["quaternions"][:])
                for piece in pieces:
                    segments.append((user_id, session_id, piece))
    return segments


class HeadPoseWindowDataset(Dataset):
    def __init__(self, h5_path, user_ids, context=CONTEXT_LENGTH, horizon=HORIZON, stride=20):
        self.context = context
        self.horizon = horizon
        self.window = context + horizon
        self.segments = load_segments(h5_path, user_ids)
        self.index = []
        for segment_id, (_, _, quaternions) in enumerate(self.segments):
            last_start = len(quaternions) - self.window
            if last_start < 0:
                continue
            for start in range(0, last_start + 1, stride):
                self.index.append((segment_id, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        segment_id, start = self.index[item]
        quaternions = self.segments[segment_id][2]
        window = quaternions[start : start + self.window]
        return torch.from_numpy(window.copy())


def context_targets(windows, context):
    """windows: (batch, context + horizon, 4) xyzw on a 100 Hz grid."""
    context_quat = windows[:, :context]
    future = windows[:, context:]
    deltas = quat_to_rotvec(quat_mul(quat_inv(context_quat[:, :-1]), context_quat[:, 1:]))
    q_last = context_quat[:, -1]
    return deltas, q_last, future


def predict_quaternions(model, windows, context):
    deltas, q_last, future = context_targets(windows, context)
    rotvecs, _hidden = model(deltas)
    return apply_body_rotvec(q_last, rotvecs), future


def official_loss(prediction, future):
    """Geodesic error. The scored figure is the angle at the 20th sample."""
    degrees = geodesic_degrees(prediction, future)
    return degrees.mean() + degrees[:, -1].mean(), degrees
