# models/generator.py

import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, seq_len=1248, latent_dim=100):
        super().__init__()

        self.seq_len = seq_len

        self.projected_len = seq_len // 8  # 1248 / 8 = 156

        self.fc = nn.Linear(latent_dim, self.projected_len * 128)

        self.net = nn.Sequential(

            # (B, 128, 156)
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # (B, 128, 312)
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # (B, 64, 624)
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Output (B, 1, 1248)
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, z):

        x = self.fc(z)
        x = x.view(z.size(0), 128, self.projected_len)

        x = self.net(x)

        return x