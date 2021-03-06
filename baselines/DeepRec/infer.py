"""Generates a valid submission file using a saved trained model"""

# Copyright (c) 2017 NVIDIA Corporation
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from data_utils.CIL_data_converter import convert2CILdictionary
from reco_encoder.data import input_layer
from reco_encoder.model import model
from torch.autograd import Variable

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
    default="",
    metavar="N",
    help="Path to training data",
)
parser.add_argument(
    "--path_to_eval_data",
    type=str,
    default="",
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
    default="autorec.pt",
    metavar="N",
    help="where to save model",
)
parser.add_argument(
    "--predictions_path",
    type=str,
    default="deeprec_predictions.csv",
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
    # Loads up training data
    params = dict()
    params["batch_size"] = 1
    params["data_dir"] = args.path_to_train_data
    params["major"] = "users"
    params["itemIdInd"] = 1
    params["userIdInd"] = 0
    print("Loading training data")
    data_layer = input_layer.UserItemRecDataProvider(params=params)
    print("Data loaded")
    print("Total items found: {}".format(len(data_layer.data.keys())))
    print("Vector dim: {}".format(data_layer.vector_dim))

    # Loads submission data as evaluation data
    print("Loading submission data")
    eval_params = copy.deepcopy(params)
    eval_params["batch_size"] = 1
    eval_params["data_dir"] = args.path_to_eval_data
    eval_data_layer = input_layer.UserItemRecDataProvider(
        params=eval_params,
        user_id_map=data_layer.userIdMap,
        item_id_map=data_layer.itemIdMap,
    )
    print("Total submission items found: {}".format(len(eval_data_layer.data.keys())))
    print("Vector dim: {}".format(eval_data_layer.vector_dim))

    # Initializes the model
    rencoder = model.AutoEncoder(
        layer_sizes=[data_layer.vector_dim]
        + [int(l_sizes) for l_sizes in args.hidden_layers.split(",")],
        nl_type=args.non_linearity_type,
        is_constrained=args.constrained,
        dp_drop_prob=args.drop_prob,
        last_layer_activations=not args.skip_last_layer_nl,
    )

    # Overrides the initialized model with a saved pre-trained model
    path_to_model = Path(args.save_path)
    if path_to_model.is_file():
        print("Loading model from: {}".format(path_to_model))
        rencoder.load_state_dict(torch.load(args.save_path))

    print("######################################################")
    print("############# AutoEncoder Model: #####################")
    print(rencoder)
    print("######################################################")

    rencoder.eval()

    if use_gpu:
        rencoder = rencoder.cuda()

    inv_userIdMap = {v: k for k, v in data_layer.userIdMap.items()}
    inv_itemIdMap = {v: k for k, v in data_layer.itemIdMap.items()}

    eval_data_layer.src_data = data_layer.data

    preds = dict()

    # Performs a complete iteration in evaluation mode on the submission data
    # generating predicted ratings
    for i, ((out, src), majorInd) in enumerate(
        eval_data_layer.iterate_one_epoch_eval(for_inf=True)
    ):
        inputs = Variable(src.cuda().to_dense() if use_gpu else src.to_dense())
        targets_np = out.to_dense().numpy()[0, :]
        outputs = rencoder(inputs).cpu().data.numpy()[0, :]
        non_zeros = targets_np.nonzero()[0].tolist()
        major_key = inv_userIdMap[majorInd]  # user
        preds[major_key] = []
        for ind in non_zeros:
            preds[major_key].append((inv_itemIdMap[ind], outputs[ind]))
        if i % 1000 == 0:
            print("Done: {} predictions".format(i))

    # Writes final submission file
    preds = convert2CILdictionary(preds)
    with open(args.predictions_path, "w") as outf:
        outf.write("Id,Prediction\n")
        for item in preds:
            for user, rating in preds[item]:
                rating = np.clip(rating, 1.0, 5.0)
                outf.write(f"r{user}_c{item},{rating}\n")


if __name__ == "__main__":
    main()
