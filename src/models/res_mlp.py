# src/models/res_mlp.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.relu(out + residual)

class ResidualMLP(nn.Module):
    def __init__(self, hidden_size=128, horizon=20, num_blocks=3, dropout=0.1, **kwargs):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        
        # Sử dụng LazyLinear để tự động nhận diện chiều dữ liệu đầu vào
        self.input_proj = nn.LazyLinear(hidden_size)
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout) for _ in range(num_blocks)
        ])
        
        self.head = None 
        self.horizon_val = horizon
        self.hidden_dim = hidden_size

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.shape[0]
        input_dim = x.shape[-1]
        flat_x = x.reshape(batch_size, -1)
        
        # Khởi tạo head dynamically trong lần forward đầu tiên
        if self.head is None:
            self.head = nn.Linear(self.hidden_dim, self.horizon_val * input_dim).to(x.device)

        h = self.input_proj(flat_x)
        for block in self.blocks:
            h = block(h)
            
        out = self.head(h)
        out = out.view(batch_size, self.horizon_val, input_dim)
        return out, None

    def count_parameters(self):
        # Tự động chạy một dummy forward để khởi tạo tham số trước khi đếm
        if any(isinstance(p, torch.nn.parameter.UninitializedParameter) for p in self.parameters()):
            with torch.no_grad():
                device = next(self.parameters()).device
                # Tạo dummy tensor với shape chuẩn (batch=1, seq_len=399, input_dim=3)
                dummy = torch.zeros(1, 399, 3, device=device)
                self.forward(dummy)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)