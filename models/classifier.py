import torch.nn as nn


class ECGClassifier(nn.Module):
    def __init__(self, seq_len=1248, n_classes=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Flatten(),
            nn.Linear(128 * (seq_len // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.net(x)