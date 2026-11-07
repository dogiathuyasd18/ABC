"""Shared decode: model rotvecs → unit quaternions."""

from src.utils.quaternions import apply_body_rotvec, quat_normalize


def decode_model(model, deltas, q_last):
    rotvecs, hidden = model(deltas)
    quaternions = quat_normalize(apply_body_rotvec(q_last, rotvecs))
    return quaternions, hidden
