import torch
from torch.utils.data import Dataset
import numpy as np

class AnonymizedDataset(Dataset):
    def __init__(self, path):
        self.data = np.loadtxt(path, delimiter=',', skiprows=1, usecols=[1, 2], dtype=np.float32)
        self.data = (self.data - self.data.mean(axis=0)) / np.std(self.data, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index][0], dtype=torch.float32)
        y = torch.tensor(self.data[index][1], dtype=torch.float32)
        return x, y

