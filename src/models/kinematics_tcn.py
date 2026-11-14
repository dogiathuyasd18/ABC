"""Kinematics-aware Temporal Convolutional Network for future head-pose rotation increments.

Input:
    deltas: [batch, context - 1, 3], body-frame rotation increments in radians/frame.

Output:
    future_rotvecs: [batch, horizon, 3], future body-frame rotation increments
                    in radians/frame, plus a `None` state placeholder for API
                    compatibility with the previous KinematicsGRU.

The model keeps the original kinematics features and constant-angular-
acceleration baseline, replaces the GRU with a causal dilated TCN, and predicts
an acceleration-conditioned 3-D residual correction.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def rotate_vector_by_rotvec(vector: torch.Tensor, rotvec: torch.Tensor) -> torch.Tensor:
    """Apply Exp(rotvec) to vector using Rodrigues' rotation formula."""
    theta2 = torch.sum(rotvec * rotvec, dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-12))

    a = torch.sin(theta) / theta
    b = (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12)

    cross1 = torch.cross(rotvec, vector, dim=-1)
    cross2 = torch.cross(rotvec, cross1, dim=-1)
    rotated = vector + a * cross1 + b * cross2

    # Stable small-angle limit.
    small_angle = vector + cross1 + 0.5 * cross2
    return torch.where(theta2 < 1e-8, small_angle, rotated)


class Chomp1d(nn.Module):
    """Remove right-side padding so a convolution remains strictly causal."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[..., :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Two-layer residual causal convolution block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.out_act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.chomp1(y)
        y = self.act1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = self.act2(y)
        y = self.drop2(y)

        return self.out_act(x + y)


class KinematicsTCN(nn.Module):
    """Kinematics-aware TCN with acceleration-conditioned residual correction.

    The kinematics calculation intentionally follows the previous KinematicsGRU:
      * omega: body-frame angular velocity (approximately rad/s)
      * alpha: transported body-frame angular acceleration (approximately rad/s^2)

    The TCN sees [delta, omega, alpha] at every context step. The correction
    head additionally receives the final omega vector, final alpha vector,
    and |alpha_last|, so its predicted 3-D residual can adapt to the current
    acceleration magnitude and direction.

    Args are deliberately close to the former KinematicsGRU. `num_blocks=7`
    gives dilations [1, 2, 4, 8, 16, 32, 64]. With two kernel-3 convolutions
    per block, the receptive field is 509 samples, covering the default
    context of 399 input deltas (from 400 poses).
    """

    def __init__(
        self,
        hidden_size: int = 128,
        horizon: int = 20,
        dt: float = 0.01,
        dropout: float = 0.1,
        residual_scale: float = 0.1,
        num_blocks: int = 7,
        kernel_size: int = 3,
        omega_scale: float = 5.0,
        alpha_scale: float = 100.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if omega_scale <= 0 or alpha_scale <= 0:
            raise ValueError("omega_scale and alpha_scale must be positive")

        self.hidden_size = int(hidden_size)
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.residual_scale = float(residual_scale)
        self.num_blocks = int(num_blocks)
        self.kernel_size = int(kernel_size)
        self.omega_scale = float(omega_scale)
        self.alpha_scale = float(alpha_scale)

        # Per-timestep kinematic feature vector: 3 + 3 + 3 = 9.
        self.input_proj = nn.Linear(9, hidden_size)
        self.input_act = nn.GELU()

        blocks = []
        dilations = [2**i for i in range(num_blocks)]
        for dilation in dilations:
            blocks.append(
                TemporalBlock(
                    channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)

        # Extra 7 values condition the correction on current motion:
        # omega_last (3), alpha_last (3), and ||alpha_last|| (1).
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_size + 7, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon * 3),
        )

        # Start from the same physics baseline as the old model. The learned
        # residual is initially zero and is learned from the training data.
        final = self.correction_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @property
    def receptive_field(self) -> int:
        """Number of input timesteps visible to the final TCN output."""
        max_dilation = sum(2**i for i in range(self.num_blocks))
        return 1 + 2 * (self.kernel_size - 1) * max_dilation

    def _kinematics(self, deltas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dt = self.dt

        # Body-frame angular velocity, approximately rad/s.
        omega = deltas / dt

        # Transport the previous angular velocity into the current body frame
        # before taking the finite difference.
        previous_in_current = rotate_vector_by_rotvec(
            omega[:, :-1],
            -deltas[:, :-1],
        )
        alpha_valid = (omega[:, 1:] - previous_in_current) / dt

        # Align acceleration sequence with omega sequence.
        alpha = torch.cat(
            [torch.zeros_like(omega[:, :1]), alpha_valid],
            dim=1,
        )
        return omega, alpha

    def forward(
        self,
        deltas: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, None]:
        # `hidden` is accepted only for call-site compatibility with KinematicsGRU.
        del hidden

        if deltas.ndim != 3 or deltas.shape[-1] != 3:
            raise ValueError(
                f"Expected deltas [B,T,3], got {tuple(deltas.shape)}"
            )
        if deltas.shape[1] < 2:
            raise ValueError("KinematicsTCN requires at least two deltas")

        omega, alpha = self._kinematics(deltas)

        features = torch.cat(
            [
                deltas,
                omega / self.omega_scale,
                alpha / self.alpha_scale,
            ],
            dim=-1,
        )

        # Linear projection over features, then TCN expects [B,C,T].
        encoded = self.input_act(self.input_proj(features))
        temporal = self.tcn(encoded.transpose(1, 2))
        last_temporal = temporal[:, :, -1]

        omega_last = omega[:, -1, :] / self.omega_scale
        alpha_last = alpha[:, -1, :] / self.alpha_scale
        alpha_magnitude = torch.linalg.vector_norm(
            alpha[:, -1, :],
            dim=-1,
            keepdim=True,
        ) / self.alpha_scale

        correction_features = torch.cat(
            [last_temporal, omega_last, alpha_last, alpha_magnitude],
            dim=-1,
        )
        residual = self.correction_head(correction_features).view(
            deltas.shape[0], self.horizon, 3
        )

        # Preserve the original constant-angular-acceleration baseline.
        steps = torch.arange(
            1,
            self.horizon + 1,
            device=deltas.device,
            dtype=deltas.dtype,
        ).view(1, self.horizon, 1)

        omega_last_physical = omega[:, -1:, :]
        alpha_last_physical = alpha[:, -1:, :]
        baseline = (
            omega_last_physical + steps * self.dt * alpha_last_physical
        ) * self.dt

        prediction = baseline + self.residual_scale * residual
        return prediction, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
