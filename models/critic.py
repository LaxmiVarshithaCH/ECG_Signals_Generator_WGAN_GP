# models/critic.py

import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, seq_len=1248):
        super().__init__()

        self.net = nn.Sequential(

            # (B, 1, 1248)
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(256 * (seq_len // 16), 1)
        )

    def forward(self, x):
        return self.net(x)