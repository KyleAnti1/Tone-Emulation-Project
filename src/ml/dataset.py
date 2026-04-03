import torch
from torch.utils.data import Dataset
import numpy as np

class AudioPairDataset(Dataset):
    def __init__(self, dry, wet, window_size):
        print ()
        assert len(dry) == len(wet)

        self.window_size = window_size
        self.half = window_size // 2

        self.dry = dry
        self.wet = wet

        self.length = len(dry) - 2 * self.half

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx + self.half

        x = self.dry[idx - self.half : idx + self.half + 1]
        y = self.wet[idx]

        return torch.tensor(x, dtype=torch.float32), \
            torch.tensor(y, dtype=torch.float32)