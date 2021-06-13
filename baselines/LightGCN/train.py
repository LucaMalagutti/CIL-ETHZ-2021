import argparse
import os

import numpy as np
import torch
from dataloader import get_dataloader
from lightGCN import LightGCN
from loss import RMSELoss
from torch import optim


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LightGCN(args)

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt))

    model.to(device)

    model.train()

    train_dataloader = get_dataloader(args, eval=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # RMSE = RMSELoss()

    for i_batch, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        print(batch.shape)
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name", type=str, help="name of the experiment", required=True
    )

    parser.add_argument(
        "--emb_size", type=int, default=64, help="Embedding size of LightGCN"
    )

    parser.add_argument(
        "--batch_size", type=int, default=2048, help="batch size of the model"
    )

    parser.add_argument(
        "--n_layer",
        type=int,
        default=3,
        help="number of iterations of the aggregation function in LightGCN",
    )

    parser.add_argument("--lr", type=float, default=0.001, help="the learning rate")

    parser.add_argument(
        "--dropout", type=float, default=0.0, help="dropout probability 0 to disable it"
    )

    parser.add_argument(
        "--epochs", type=int, default=1000, help="number of training epochs"
    )

    parser.add_argument(
        "--restore_ckpt",
        type=str,
        default=None,
        help="restore model weights from checkpoint",
    )

    args = parser.parse_args()

    torch.manual_seed(2021)
    np.random.seed(2021)

    if not os.path.isdir("checkpoints"):
        os.mkdir("checkpoints")

    train(args)
