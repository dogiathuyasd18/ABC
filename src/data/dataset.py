"""User-disjoint windows from resampled head-pose segments.

HDF5 stores sequences. This module only builds a window index: 400 context
samples plus 20 targets, never crossing a session or a gap cut.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.prep.config import CONTEXT_SAMPLES, HORIZON_SAMPLES, TARGET_SAMPLE_RATE_HZ, WindowConfig
from src.prep.hdf5 import iter_segments, list_users as hdf5_list_users
from src.prep.resampling import resample_session
from src.prep.windows import index_segment
from src.utils.quaternions import apply_body_rotvec, geodesic_degrees, quat_inv, quat_mul, quat_to_rotvec

CONTEXT_LENGTH = CONTEXT_SAMPLES
HORIZON = HORIZON_SAMPLES
SAMPLE_RATE_HZ = TARGET_SAMPLE_RATE_HZ
MIN_WINDOW = CONTEXT_SAMPLES + HORIZON_SAMPLES


def list_users(h5_path):
    return hdf5_list_users(Path(h5_path))


def split_users(user_ids, val_fraction=0.2):
    if not user_ids:
        raise ValueError("No users in the HDF5 file")
    val_count = max(1, int(round(len(user_ids) * val_fraction)))
    if val_count >= len(user_ids):
        val_count = 1
    return user_ids[:-val_count], user_ids[-val_count:]


def resample_session_legacy(timestamps, quaternions):
    """Kept for old session-level HDF5 rows. Uses the documented gap threshold."""
    result = resample_session(timestamps, quaternions)
    return [segment.quaternions for segment in result.segments]


def load_segments(h5_path, user_ids):
    segments = []
    for item in iter_segments(Path(h5_path), user_ids):
        attrs = item.get("session_attrs") or {}
        already_resampled = bool(attrs.get("resampled", item["segment_id"] != "legacy"))
        if item["segment_id"] == "legacy" and not already_resampled:
            result = resample_session(item["timestamps"], item["quaternions"])
            for index, piece in enumerate(result.segments):
                segments.append((item["user_id"], item["session_id"], f"legacy-{index:04d}", piece.quaternions))
        else:
            segments.append((item["user_id"], item["session_id"], item["segment_id"], item["quaternions"]))
    return segments


class HeadPoseWindowDataset(Dataset):
    def __init__(self, h5_path, user_ids, context=CONTEXT_LENGTH, horizon=HORIZON, stride=20):
        self.context = context
        self.horizon = horizon
        self.window = context + horizon
        cfg = WindowConfig(context_samples=context, horizon_samples=horizon, stride=stride)
        self.segments = load_segments(h5_path, user_ids)
        self.index = []
        for segment_id, (user_id, session_id, name, quaternions) in enumerate(self.segments):
            for window in index_segment(user_id, session_id, name, len(quaternions), cfg):
                self.index.append((segment_id, window.start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        segment_id, start = self.index[item]
        quaternions = self.segments[segment_id][3]
        window = quaternions[start : start + self.window]
        if len(window) != self.window:
            raise RuntimeError("Window shorter than context+horizon")
        return torch.from_numpy(window.copy())


def context_targets(windows, context):
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
    degrees = geodesic_degrees(prediction, future)
    return degrees.mean() + degrees[:, -1].mean(), degrees
