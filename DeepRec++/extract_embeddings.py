"""
    Extracts an inner intermediate representation (embedding) of the user or items vectors
    from a saved autoencoder model
"""
# Copyright (c) 2017 NVIDIA Corporation
import argparse
import copy
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data_utils.CIL_data_converter import convert2CILdictionary
from reco_encoder.data import input_layer
from reco_encoder.model import model
from torch.autograd import Variable

torch.manual_seed(44)
random.seed(44)
np.random.seed(44)

parser = argparse.ArgumentParser(description="RecoEncoder")

parser.add_argument(
    "--drop_prob", type=float, default=0.0, metavar="N", help="dropout drop probability"
)
parser.add_argument(
    "--constrained", action="store_true", help="constrained autoencoder"
)
parser.add_argument(
    "--skip_last_layer_nl",
    action="store_true",
    help="if present, decoder's last layer will not apply non-linearity function",
)
parser.add_argument(
    "--hidden_layers",
    type=str,
    default="1024,512,512,128",
    metavar="N",
    help="hidden layer sizes, comma-separated",
)
parser.add_argument(
    "--path_to_train_data",
    type=str,
    default="data/train90",
    metavar="N",
    help="Path to training data",
)
parser.add_argument(
    "--path_to_eval_data",
    type=str,
    default="data/valid",
    metavar="N",
    help="Path to evaluation data",
)
parser.add_argument(
    "--non_linearity_type",
    type=str,
    default="selu",
    metavar="N",
    help="type of the non-linearity used in activations",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="model_save/pretrain_emb/",
    metavar="N",
    help="where to load saved model from",
)
parser.add_argument(
    "--predictions_path",
    type=str,
    default="out.json",
    metavar="N",
    help="where to save predictions",
)
parser.add_argument(
    "--major",
    type=str,
    default="users",
    metavar="N",
    help="produce user-based or item-based autoencoder",
)

args = parser.parse_args()
print(args)

use_gpu = torch.cuda.is_available()  # global flag
if use_gpu:
    print("GPU is available.")
else:
    print("GPU is not available.")


def main():
    params = dict()
    params["batch_size"] = 1
    params["data_dir"] = args.path_to_train_data
    params["major"] = args.major
    params["itemIdInd"] = 1
    params["userIdInd"] = 0
    print("Loading training data")
    data_layer = input_layer.UserItemRecDataProvider(params=params)
    print("Data loaded")
    print("Total items found: {}".format(len(data_layer.data.keys())))
    print("Vector dim: {}".format(data_layer.vector_dim))

    Path(args.predictions_path).mkdir(parents=True, exist_ok=True)

    print("Loading submission data")
    eval_params = copy.deepcopy(params)
    # must set eval batch size to 1 to make sure no examples are missed
    eval_params["batch_size"] = 1
    eval_params["data_dir"] = args.path_to_eval_data
    eval_data_layer = input_layer.UserItemRecDataProvider(
        params=eval_params,
        user_id_map=data_layer.userIdMap,
        item_id_map=data_layer.itemIdMap,
    )
    print("Total submission items found: {}".format(len(eval_data_layer.data.keys())))
    print("Vector dim: {}".format(eval_data_layer.vector_dim))

    rencoder = model.AutoEncoder(
        layer_sizes=[data_layer.vector_dim]
        + [int(l_sizes) for l_sizes in args.hidden_layers.split(",")],
        nl_type=args.non_linearity_type,
        is_constrained=args.constrained,
        dp_drop_prob=args.drop_prob,
        last_layer_activations=not args.skip_last_layer_nl,
        extract_embeddings=True,
    )

    path_to_model = Path(args.save_path)
    if path_to_model.is_file():
        print("Loading model from: {}".format(path_to_model))
        rencoder.load_state_dict(torch.load(args.save_path))

    print("######################################################")
    print("######################################################")
    print("############# AutoEncoder Model: #####################")
    print(rencoder)
    print("######################################################")
    print("######################################################")

    rencoder.eval()

    if use_gpu:
        rencoder = rencoder.cuda()

    inv_userIdMap = {v: k for k, v in data_layer.userIdMap.items()}
    inv_itemIdMap = {v: k for k, v in data_layer.itemIdMap.items()}

    eval_data_layer.src_data = data_layer.data

    embeddings_dict = dict()

    # Cycle through all the major elements (users or items) in the eval dataset and feed it to the autoencoder,
    # saving the state of the central layer of the autoencoder as an embedding for each
    for i, ((out, src), majorInd) in enumerate(
        eval_data_layer.iterate_one_epoch_eval(for_inf=True)
    ):

        # input for the autoencoder
        # for the users it's the 1000 sized vector of their ratings
        # similarly for the items it's a 10000 sized vector
        major_ratings = Variable(src.cuda().to_dense() if use_gpu else src.to_dense())

        # rencoder is built to output the deep features
        embedding = rencoder(major_ratings).cpu().data.numpy()[0, :]

        # get user idx as in the train data
        if args.major == "users":
            major_idx = inv_userIdMap[majorInd]
        else:
            major_idx = inv_itemIdMap[majorInd]

        if major_idx not in embeddings_dict:

            # new user: save deep features in the dictionary
            embeddings_dict[int(major_idx)] = embedding.tolist()

            if i % 500 == 0:
                print("Extracted deep features for {} major".format(i))

    print("Saving embedding dictionary..")
    print(len(embeddings_dict))
    emb_path = args.predictions_path + args.major + "_emb.pckl"
    with open(emb_path, "wb") as outf:
        pickle.dump(embeddings_dict, outf)

    print("Saved embedding dictionary to:", args.predictions_path)


if __name__ == "__main__":
    main()
