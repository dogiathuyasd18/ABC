"""Causal GRU that maps recent head motion to 20 future body-frame rotations.

The hidden state is a function of past samples only. At inference the same
weights can be stepped one sample at a time.
"""

import torch
from torch import nn


class HeadPoseGRU(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, horizon=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.gru = nn.GRU(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, horizon * 3)

    def forward(self, delta_rotvec, hidden=None):
        """delta_rotvec: (batch, time, 3) body-frame steps. Returns (batch, horizon, 3)."""
        output, hidden = self.gru(delta_rotvec, hidden)
        prediction = self.head(output[:, -1])
        return prediction.view(prediction.shape[0], self.horizon, 3), hidden

    def count_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
