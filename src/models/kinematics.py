
"""Kinematics-aware GRU for future head-pose rotation increments."""

import torch
from torch import nn


def rotate_vector_by_rotvec(vector, rotvec):
    """Apply Exp(rotvec) to vector using Rodrigues' rotation formula."""
    theta2 = torch.sum(rotvec * rotvec, dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-12))

    a = torch.sin(theta) / theta
    b = (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12)

    cross1 = torch.cross(rotvec, vector, dim=-1)
    cross2 = torch.cross(rotvec, cross1, dim=-1)

    rotated = vector + a * cross1 + b * cross2

    # Stable small-angle limit.
    return torch.where(theta2 < 1e-8, vector + cross1 + 0.5 * cross2, rotated)


class KinematicsGRU(nn.Module):
    """
    Predict future body-frame rotation increments.

    Input:
        deltas: [batch, context-1, 3], radians per frame.

    Output:
        future_rotvecs: [batch, horizon, 3], radians per future frame.
    """

    def __init__(
        self,
        hidden_size=128,
        num_layers=2,
        horizon=20,
        dt=0.01,
        dropout=0.0,
        residual_scale=0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.dt = dt
        self.residual_scale = residual_scale

        self.input_proj = nn.Linear(9, hidden_size)

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Linear(hidden_size, horizon * 3)

        # Initially use the kinematic baseline without a learned correction.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _kinematics(self, deltas):
        dt = self.dt

        # Body-frame angular velocity, approximately rad/s.
        omega = deltas / dt

        # Transport the previous angular velocity into the current
        # body frame before taking the finite difference.
        previous_in_current = rotate_vector_by_rotvec(
            omega[:, :-1],
            -deltas[:, :-1],
        )

        alpha_valid = (omega[:, 1:] - previous_in_current) / dt

        # Align acceleration sequence with omega sequence.
        alpha = torch.cat(
            [
                torch.zeros_like(omega[:, :1]),
                alpha_valid,
            ],
            dim=1,
        )

        return omega, alpha

    def forward(self, deltas, hidden=None):
        if deltas.ndim != 3 or deltas.shape[-1] != 3:
            raise ValueError(
                f"Expected deltas [B,T,3], got {tuple(deltas.shape)}"
            )

        if deltas.shape[1] < 2:
            raise ValueError("KinematicsGRU requires at least two deltas")

        omega, alpha = self._kinematics(deltas)

        # Feature scaling keeps the three physical quantities
        # in a more manageable numerical range.
        features = torch.cat(
            [
                deltas,
                omega / 5.0,
                alpha / 100.0,
            ],
            dim=-1,
        )

        encoded = torch.tanh(self.input_proj(features))
        output, hidden = self.gru(encoded, hidden)

        residual = self.head(output[:, -1]).view(
            deltas.shape[0], self.horizon, 3
        )

        # Constant-angular-acceleration baseline.
        steps = torch.arange(
            1,
            self.horizon + 1,
            device=deltas.device,
            dtype=deltas.dtype,
        ).view(1, self.horizon, 1)

        omega_last = omega[:, -1:, :]
        alpha_last = alpha[:, -1:, :]

        baseline = (
            omega_last + steps * self.dt * alpha_last
        ) * self.dt

        prediction = baseline + self.residual_scale * residual

        return prediction, hidden

    def count_parameters(self):
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
