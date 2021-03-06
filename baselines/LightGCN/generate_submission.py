"""Generates a submission file given a pretrained saved model"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from dataloader import get_dataloader
from lightGCN import LightGCN
from tqdm import tqdm


def generate_submission(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loads the hyperparameters of the saved model
    saved_state_dict = torch.load(args.restore_ckpt)
    args.lr = saved_state_dict["lr"]
    args.emb_size = saved_state_dict["emb_size"]
    args.n_layers = saved_state_dict["n_layers"]
    args.batch_size = saved_state_dict["batch_size"]
    # Instantiates and loads the saved model
    model = LightGCN(args)
    model.load_state_dict(saved_state_dict["model_state_dict"])

    model.to(device)

    model.eval()
    # Creates submission dataset
    test_dataloader = get_dataloader(args, split="test", shuffle=False)

    sub_data = pd.read_csv("data/sub.csv")
    scores_list = []

    # Performs rating prediction for all submission pairs
    for batch in tqdm(test_dataloader):
        if torch.cuda.is_available():
            batch = batch.cuda()
        scores_list.extend(model(batch[:, :2]).tolist())

    assert len(scores_list) == len(sub_data["rating"])
    # Saves predictions to a submission file
    sub_data["Prediction"] = scores_list
    sub_data["Id"] = (
        "r"
        + sub_data["user"].apply(lambda x: str(x + 1))
        + "_"
        + "c"
        + sub_data["movie"].apply(lambda x: str(x + 1))
    )
    sub_data = sub_data[["Id", "Prediction"]]

    sub_name = args.restore_ckpt.split("_")[1].split(".")[0] + ".csv"
    sub_data.to_csv(os.path.join("submissions", sub_name), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--restore_ckpt",
        type=str,
        default=None,
        help="restore model weights from checkpoint",
        required=True,
    )

    args = parser.parse_args()

    torch.manual_seed(2021)
    np.random.seed(2021)

    if not os.path.isdir("submissions"):
        os.mkdir("submissions")

    generate_submission(args)
