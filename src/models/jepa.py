# src/models/jepa.py
import torch
import torch.nn as nn

class HeadPoseJEPA(nn.Module):
    def __init__(self, input_dim=3, latent_dim=128, horizon=20, hidden_size=256, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        
        # 1. Context Encoder (GrU/Encoder backbone)
        self.context_encoder = nn.GRU(
            input_size=input_dim, 
            hidden_size=latent_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=dropout
        )
        
        # 2. Predictor (dự đoán trong không gian ẩn tương lai)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, latent_dim * horizon)
        )
        
        # 3. Projector/Decoder đưa về không gian output gốc
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.latent_dim = latent_dim

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]
        
        # Encode context
        _, h_n = self.context_encoder(x) # h_n shape: (num_layers, batch_size, latent_dim)
        context_latent = h_n[-1] # Lấy tầng cuối: (batch_size, latent_dim)
        
        # Dự đoán latent đại diện cho các bước tương lai
        pred_latents = self.predictor(context_latent) # (batch_size, horizon * latent_dim)
        pred_latents = pred_latents.view(batch_size, self.horizon, self.latent_dim)
        
        # Decode về output không gian gốc
        out = self.decoder(pred_latents) # (batch_size, horizon, input_dim)
        
        return out, None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)