"""Contains training and evaluation procedures for the model"""

import argparse
import os

import numpy as np
import torch
import wandb
from dataloader import get_dataloader
from lightGCN import LightGCN
from loss import RMSELoss
from torch import optim


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiates the model
    model = LightGCN(args)
    # Loads a saved model to continue training, if given
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt))
    wandb.watch(model)

    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    RMSE = RMSELoss()
    # Loads training data
    train_dataloader = get_dataloader(args, split="train")

    training_loss = 0.0
    for i_epoch in range(args.epochs):
        print(f"Starting epoch: {i_epoch}")
        # Trains for an epoch
        for i_batch, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            if torch.cuda.is_available():
                batch = batch.cuda()

            scores = model(batch[:, :2])
            loss = RMSE(scores, batch[:, 2])
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            # Prints epoch summary RMSE loss
            if i_batch % PRT_FREQ == (PRT_FREQ - 1):
                print(f"LightGCN_train_loss {training_loss / PRT_FREQ}")
                wandb.log({"LightGCN_train_loss": training_loss / PRT_FREQ})
                training_loss = 0.0

        if i_epoch % EVAL_FREQ == (EVAL_FREQ - 1):
            evaluate(args, model)

            # Saves the model and its hyperparams in the checkpoints/ folder after evaluation
            PATH = f"checkpoints/{i_epoch+1}_{args.name}.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "n_layers": args.n_layers,
                    "emb_size": args.emb_size,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                },
                PATH,
            )


def evaluate(args, model):
    # Evaluates the model on the val dataset
    model.eval()
    evaluate_dataloader = get_dataloader(args, split="eval")

    RMSE = RMSELoss()

    rmse_eval = 0.0

    for _, batch in enumerate(evaluate_dataloader):
        if torch.cuda.is_available():
            batch = batch.cuda()

        scores = model(batch[:, :2])
        loss = RMSE(scores, batch[:, 2])
        rmse_eval += loss.item()

    rmse_eval /= len(evaluate_dataloader)
    print(f"LightGCN_evaluation_RMSE: {rmse_eval}\n")
    wandb.log({"LightGCN_RMSE": rmse_eval})


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
        "--n_layers",
        type=int,
        default=3,
        help="number of iterations of the aggregation function in LightGCN",
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument(
        "--restore_ckpt",
        type=str,
        default=None,
        help="restore model weights from checkpoint",
    )

    args = parser.parse_args()
    wandb.init(project="CIL-2021", entity="spaghetticode")

    torch.manual_seed(2021)
    np.random.seed(2021)

    if not os.path.isdir("checkpoints"):
        os.mkdir("checkpoints")

    PRT_FREQ = 100
    EVAL_FREQ = 5

    train(args)
