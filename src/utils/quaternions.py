"""Quaternion helpers in xyzw order. Model code stays in PyTorch so it exports."""

import torch


def quat_normalize(q, eps=1e-8):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_inv(q):
    xyz, w = q[..., :3], q[..., 3:]
    return torch.cat((-xyz, w), dim=-1)


def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(dim=-1)
    bx, by, bz, bw = b.unbind(dim=-1)
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack((x, y, z, w), dim=-1)


def quat_to_rotvec(q):
    q = quat_normalize(q)
    xyz = q[..., :3]
    w = q[..., 3].clamp(-1.0, 1.0)
    sin_half = xyz.norm(dim=-1)
    half = torch.atan2(sin_half, w)
    angle = 2.0 * half
    scale = torch.where(
        sin_half > 1e-6,
        angle / sin_half.clamp_min(1e-8),
        torch.full_like(sin_half, 2.0),
    )
    return xyz * scale.unsqueeze(-1)


def rotvec_to_quat(rotvec):
    angle = rotvec.norm(dim=-1, keepdim=True)
    half = angle * 0.5
    small = angle < 1e-6
    axis = rotvec / angle.clamp_min(1e-8)
    xyz = torch.where(small, rotvec * 0.5, axis * torch.sin(half))
    w = torch.where(small.squeeze(-1), torch.ones_like(half.squeeze(-1)), torch.cos(half.squeeze(-1)))
    return quat_normalize(torch.cat((xyz, w.unsqueeze(-1)), dim=-1))


def apply_body_rotvec(q_last, rotvecs):
    """Right-multiply body-frame rotations onto the last context quaternion."""
    delta = rotvec_to_quat(rotvecs)
    base = q_last.unsqueeze(-2).expand_as(delta)
    return quat_normalize(quat_mul(base, delta))


def geodesic_degrees(q_hat, q):
    """Mean-ready geodesic angle in degrees. Uses |<q_hat, q>| for double cover."""
    q_hat = quat_normalize(q_hat)
    q = quat_normalize(q)
    dot = (q_hat * q).sum(dim=-1).abs().clamp(0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(dot))
