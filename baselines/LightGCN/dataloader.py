import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CIL(Dataset):
    def __init__(self, eval=False, path="data/"):
        self.eval = eval

        self.train_df = np.loadtxt(
            open(os.path.join(path, "train.csv"), "rb"), delimiter=",", skiprows=1
        )
        self.val_df = np.loadtxt(
            open(os.path.join(path, "val.csv"), "rb"), delimiter=",", skiprows=1
        )

    def __getitem__(self, idx):
        if self.eval:
            return torch.from_numpy(self.val_df[idx])
        else:
            return torch.from_numpy(self.train_df[idx])

    def __len__(self):
        if self.eval:
            return self.val_df.shape[0]
        else:
            return self.train_df.shape[0]


def get_dataloader(args, eval=False):
    dataset = CIL(eval)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    print("Loading data with %d samples" % len(dataset))
    return dataloader
