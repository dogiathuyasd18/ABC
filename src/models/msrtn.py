
"""Multi-Scale Residual Temporal Network for head-pose forecasting."""

import torch
from torch import nn


class MultiScaleResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()

        self.conv3 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1
        )
        self.conv7 = nn.Conv1d(
            channels, channels, kernel_size=7, padding=3
        )
        self.conv15 = nn.Conv1d(
            channels, channels, kernel_size=15, padding=7
        )

        self.fusion = nn.Conv1d(channels * 3, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        multi_scale = torch.cat(
            [
                self.conv3(x),
                self.conv7(x),
                self.conv15(x),
            ],
            dim=1,
        )

        x = self.fusion(multi_scale)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)

        return residual + x


class MultiScaleResidualTemporalNetwork(nn.Module):
    def __init__(
        self,
        d_model=64,
        num_blocks=3,
        horizon=20,
        dropout=0.1,
    ):
        super().__init__()

        if d_model < 4:
            raise ValueError("d_model must be at least 4")

        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")

        if horizon < 1:
            raise ValueError("horizon must be positive")

        self.d_model = d_model
        self.num_blocks = num_blocks
        self.horizon = horizon

        self.input_projection = nn.Linear(3, d_model)

        self.blocks = nn.Sequential(
            *[
                MultiScaleResidualBlock(d_model, dropout)
                for _ in range(num_blocks)
            ]
        )

        self.attention = nn.Linear(d_model, 1)

        self.forecast_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon * 3),
        )

    def forward(self, delta_rotvec, hidden=None):
        if delta_rotvec.ndim != 3 or delta_rotvec.shape[-1] != 3:
            raise ValueError(
                "Expected input shape (batch, time, 3), "
                f"got {tuple(delta_rotvec.shape)}"
            )

        x = self.input_projection(delta_rotvec)
        x = x.transpose(1, 2)

        x = self.blocks(x)
        x = x.transpose(1, 2)

        attention_weights = torch.softmax(
            self.attention(x).squeeze(-1),
            dim=1,
        )

        context = torch.sum(
            x * attention_weights.unsqueeze(-1),
            dim=1,
        )

        prediction = self.forecast_head(context)
        prediction = prediction.reshape(
            prediction.shape[0],
            self.horizon,
            3,
        )

        return prediction, None

    def count_parameters(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
