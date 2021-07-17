import argparse
import os
import pickle
import random

import numpy as np
import pandas
import torch
import wandb
from datasets import EmbeddingDataset
from reco_encoder.model import model
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser(description="MLP")
parser.add_argument(
    "--learning_rate", type=float, default=0.00001, metavar="N", help="learning rate"
)
parser.add_argument(
    "--dropout", type=float, default=0.0, metavar="N", help="dropout drop probability"
)
parser.add_argument(
    "--num_epochs", type=int, default=50, metavar="N", help="maximum number of epochs"
)
parser.add_argument(
    "--save_every",
    type=int,
    default=5,
    metavar="N",
    help="save every N number of epochs",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="momentum",
    metavar="N",
    help="optimizer kind: adam, momentum, adagrad or rmsprop",
)
parser.add_argument(
    "--input_size",
    type=int,
    default="128",
    metavar="N",
    help="size of the input",
)
parser.add_argument(
    "--layer_sizes",
    type=str,
    default="64,32,16",
    metavar="N",
    help="hidden layer sizes",
)
parser.add_argument(
    "--gpu_ids",
    type=str,
    default="0",
    metavar="N",
    help="comma-separated gpu ids to use for data parallel training",
)
parser.add_argument(
    "--path_to_user_embs",
    type=str,
    default="data/embs/users_emb.pckl",
    metavar="N",
    help="Path to user emb",
)

parser.add_argument(
    "--path_to_item_embs",
    type=str,
    default="data/embs/items_emb.pckl",
    metavar="N",
    help="Path to item emb",
)
parser.add_argument(
    "--path_to_train_data",
    type=str,
    default="data/train90/CIL_data90.train",
    metavar="N",
    help="Path to training data",
)
parser.add_argument(
    "--path_to_eval_data",
    type=str,
    default="data/valid/CIL_data10.valid",
    metavar="N",
    help="Path to evaluation data",
)
parser.add_argument(
    "--logdir",
    type=str,
    default="model_save/mlp/",
    metavar="N",
    help="where to save model and write logs",
)

args = parser.parse_args()


def main():

    args.layer_sizes = [int(size) for size in args.layer_sizes.split(",")]
    wandb.init(
        project="CIL-2021",
        entity="spaghetticode",
        config={
            "input_size": args.input_size,
            "layer_sizes": args.layer_sizes,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
        },
    )

    mlp = model.MLP(args.input_size, args.layer_sizes, args.dropout)

    print(mlp)

    if torch.cuda.is_available():
        mlp.cuda()

    mlp.train()

    optimizer = optim.Adam(mlp.parameters(), lr=args.learning_rate)
    criterion = model.RMSELoss()

    train_dataset = EmbeddingDataset(
        args.path_to_user_embs,
        args.path_to_item_embs,
        args.path_to_train_data,
        shuffle=True,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=512)
    val_dataset = EmbeddingDataset(
        args.path_to_user_embs,
        args.path_to_item_embs,
        args.path_to_eval_data,
        shuffle=True,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=512)

    for i in range(1, args.num_epochs + 1):
        mlp.train()
        print("Epoch nr. " + str(i))
        sum_loss = 0
        for x, y in tqdm(train_dataloader):
            optimizer.zero_grad()
            y_hat = mlp(x.float())[..., 0]

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
        train_loss = sum_loss / len(train_dataloader)

        print(f"Train Loss: {train_loss}")

        wandb.log({"mlp_train_RMSE": train_loss}, step=i)

        if i % args.save_every == 0:
            mlp.eval()
            eval_loss = 0
            for x, y in tqdm(val_dataloader):
                y_hat = mlp(x.float())[..., 0]

                loss = criterion(y_hat, y)

                eval_loss += loss.item()
            eval_loss = eval_loss / len(val_dataloader)

            print(f"Eval Loss: {eval_loss}")
            wandb.log({"mlp_val_RMSE": eval_loss}, step=i)

            torch.save(mlp.state_dict(), args.logdir + "mlp@epoch_" + str(i))


if __name__ == "__main__":
    main()
