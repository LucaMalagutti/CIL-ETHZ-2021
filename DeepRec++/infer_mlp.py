"""Generates a valid submission file using a saved trained mlp and embeddings for users and items"""
import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import EmbeddingDataset
from reco_encoder.model import model
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description="RecoEncoder")

parser.add_argument(
    "--dropout", type=float, default=0.4, metavar="N", help="dropout drop probability"
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
    "--path_to_sub_data",
    type=str,
    default="data/submission/CIL_data.submission",
    metavar="N",
    help="Path to submission data",
)
parser.add_argument(
    "--logdir",
    type=str,
    default="model_save/mlp/model.last",
    metavar="N",
    help="where to load model from",
)
parser.add_argument(
    "--predictions_path",
    type=str,
    default="pred.csv",
    metavar="N",
    help="where to save predictions",
)

args = parser.parse_args()
print(args)

use_gpu = torch.cuda.is_available()  # global flag
if use_gpu:
    print("GPU is available.")
else:
    print("GPU is not available.")


def main():

    args.layer_sizes = [int(size) for size in args.layer_sizes.split(",")]

    mlp = model.MLP(args.input_size, args.layer_sizes, args.dropout)

    path_to_model = Path(args.logdir)
    # if path_to_model.is_file():
    print("Loading model from: {}".format(path_to_model))
    mlp.load_state_dict(torch.load(args.logdir))

    if use_gpu:
        mlp.cuda()

    mlp.eval()

    sub_dataset = EmbeddingDataset(
        args.path_to_user_embs, args.path_to_item_embs, args.path_to_sub_data
    )
    sub_dataloader = DataLoader(sub_dataset, batch_size=1024)

    preds = []
    for x, _ in tqdm(sub_dataloader):
        pred_batch = mlp(x)
        preds.extend([cur.item() for cur in pred_batch])

    _COL_NAMES = ["user_id", "item_id", "rating"]
    sub_data = pd.read_csv(args.path_to_sub_data, delimiter="\t", names=_COL_NAMES)

    sub_data["Prediction"] = preds
    sub_data["Id"] = (
        "r"
        + sub_data["user_id"].apply(lambda x: str(x))
        + "_"
        + "c"
        + sub_data["item_id"].apply(lambda x: str(x))
    )
    sub_data = sub_data[["Id", "Prediction"]]

    sub_data.to_csv(args.predictions_path, index=False)


if __name__ == "__main__":
    main()
