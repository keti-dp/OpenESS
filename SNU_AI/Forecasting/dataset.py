import os
import glob
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset

SEC = 86400
MEAN, STD = 0.2866035067802633, 0.3390546398502909
"""
We only use MEAN but leave STD for later usage.
"""

def compute_mean_std(npy_paths, chan=2):
    """위에 계산 값 있음."""
    import numpy as np
    s = 0.0
    ss = 0.0
    n = 0
    for p in npy_paths:
        a = np.load(p, mmap_mode='r')
        x = a[:, chan]               # [T]
        s  += float(x.sum())
        ss += float((x * x).sum())
        n  += x.shape[0]
    mean = s / n
    var  = max(ss / n - mean * mean, 0.0)
    std  = var ** 0.5
    return mean, std

class Normalize:
    def __init__(self, mean: float = MEAN, scale=1.):
        self.mean = float(mean)
        self.scale = scale

    def __call__(self, data: torch.Tensor):
        return (data - self.mean) * self.scale

    # 복원
    def inverse(self, x: torch.Tensor):
        return x / (self.scale) + self.mean
    
class Generation_Dataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True, start=55000, end=80000):

        self.start = start
        self.end = end

        # read data
        self.data = glob.glob(data_dir+'/*/*.npy')
        self.data.sort()

        if train:
            self.data = self.data[:-1]
        else:
            self.data = self.data[-3:-1]

        self.transform = transform


    def __len__(self):
        # intensity = 0 ~ 1.0
        return (len(self.data)-1) * 10

    def __getitem__(self, idx):
        date, intensity = divmod(idx, 10)

        # print(idx, date, intensity)
        arr_today = np.load(self.data[date],   mmap_mode='r')
        arr_tom   = np.load(self.data[date+1], mmap_mode='r')

        if intensity == 0:
            sl_today = slice(0, SEC)
            sl_tom   = slice(0, SEC)
        else:
            sl_today = slice(intensity*SEC, (intensity+1)*SEC)
            sl_tom   = slice((intensity+1)*SEC, (intensity+2)*SEC)

        # [T, 1] 모양을 유지하기 위해 2:3로 슬라이스
        today_np    = arr_today[sl_today, 2:3].astype(np.float32, copy=True)
        tomorrow_np = arr_tom[sl_tom,     2:3].astype(np.float32, copy=True)


        # 임의로 방전구간을 정해서 잘라 쓰겠음
        today_np    = today_np[self.start:self.end]
        tomorrow_np = tomorrow_np[self.start:self.end]

        today    = torch.from_numpy(today_np)      # [T, 1]
        tomorrow = torch.from_numpy(tomorrow_np)   # [T, 1]

        if self.transform is not None:
            today    = self.transform(today)
            tomorrow = self.transform(tomorrow)

        return today, tomorrow


def build_loader(train_dir, test_dir, batch_size):
    train_dataset = Generation_Dataset(train_dir, train=True, transform=Normalize())
    test_dataset  = Generation_Dataset(test_dir,  train=False, transform=Normalize())

    nw = 4
    common = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=False,   
        prefetch_factor=2,          
        drop_last=False,            # 긴 시퀀스면 True 권장
        # multiprocessing_context="spawn"  # 문제시 명시, 아니면 일단 생략
    )
    train_loader = DataLoader(train_dataset, **common)
    test_loader  = DataLoader(test_dataset,  **common)
    return train_loader, test_loader