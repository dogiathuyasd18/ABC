# src/models/tcn.py
import torch
import torch.nn as nn

class TemporalCNN(nn.Module):
    def __init__(self, input_dim=3, hidden_size=64, horizon=20, num_layers=4, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        
        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                nn.Conv1d(
                    in_channels, hidden_size, 
                    kernel_size=3, padding=dilation, dilation=dilation
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = hidden_size
            
        self.conv_stack = nn.Sequential(*layers)
        self.project = nn.Linear(hidden_size, input_dim * horizon)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        x_perm = x.permute(0, 2, 1)
        features = self.conv_stack(x_perm) # (batch_size, hidden_size, seq_len)
        
        # Lấy đặc trưng ở bước thời gian cuối cùng
        last_feat = features.permute(0, 2, 1)[:, -1, :] # (batch_size, hidden_size)
        
        out = self.project(last_feat) # (batch_size, horizon * input_dim)
        out = out.view(-1, self.horizon, self.input_dim)
        return out, None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)