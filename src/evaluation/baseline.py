"""Parameter-free baselines for the 20-sample (0.2 s) horizon."""

import torch

from src.utils.quaternions import apply_body_rotvec, quat_inv, quat_mul, quat_normalize, quat_to_rotvec


def last_pose_quaternions(windows, context, horizon):
    """Persistence: copy the last context quaternion for every future step."""
    q_last = quat_normalize(windows[:, context - 1])
    return q_last.unsqueeze(1).expand(-1, horizon, -1).contiguous()


def constant_velocity_quaternions(windows, context, horizon):
    """Repeat the last body-frame step. Sample k is that step times k."""
    context_quat = windows[:, :context]
    q_prev = context_quat[:, -2]
    q_last = context_quat[:, -1]
    step = quat_to_rotvec(quat_mul(quat_inv(q_prev), q_last))
    scales = torch.arange(1, horizon + 1, device=windows.device, dtype=windows.dtype)
    rotvecs = step.unsqueeze(1) * scales.view(1, horizon, 1)
    return apply_body_rotvec(q_last, rotvecs)
