# data/dataset.py

import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):

    def __init__(self, ecg_array):
        self.data = torch.from_numpy(ecg_array)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = sample.permute(1, 0)  # (1, 1248)
        return sample