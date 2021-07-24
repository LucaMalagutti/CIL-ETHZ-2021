"""Defines a Pytorch Dataset and Dataloader to use the CIL dataset"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CIL(Dataset):
    # dataset of (user_index, item_index, rating) tuples

    def __init__(self, split="train", path="data/"):
        self.split = split
        self.train_df = np.loadtxt(
            open(os.path.join(path, "train.csv"), "rb"),
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
        )
        self.val_df = np.loadtxt(
            open(os.path.join(path, "val.csv"), "rb"),
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
        )
        self.test_df = np.loadtxt(
            open(os.path.join(path, "sub.csv"), "rb"),
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
        )

    def __getitem__(self, idx):
        if self.split == "eval":
            return torch.from_numpy(self.val_df[idx])
        elif self.split == "test":
            return torch.from_numpy(self.test_df[idx])

        return torch.from_numpy(self.train_df[idx])

    def __len__(self):
        if self.split == "eval":
            return self.val_df.shape[0]
        elif self.split == "test":
            return self.test_df.shape[0]

        return self.train_df.shape[0]


def get_dataloader(args, split="train"):
    # wraps the CIL dataset into a Dataloader and returns it

    dataset = CIL(split)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )

    print("Loading data with %d samples" % len(dataset))
    return dataloader
