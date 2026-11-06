"""Constant-velocity extrapolation over the 100 ms horizon.

The last body-frame step is repeated. Sample k (1..20) is that step times k.
"""

import torch

from src.utils.quaternions import apply_body_rotvec, quat_inv, quat_mul, quat_to_rotvec


def constant_velocity_quaternions(windows, context, horizon):
    context_quat = windows[:, :context]
    q_prev = context_quat[:, -2]
    q_last = context_quat[:, -1]
    step = quat_to_rotvec(quat_mul(quat_inv(q_prev), q_last))
    scales = torch.arange(1, horizon + 1, device=windows.device, dtype=windows.dtype)
    rotvecs = step.unsqueeze(1) * scales.view(1, horizon, 1)
    return apply_body_rotvec(q_last, rotvecs)
