"""
    Extracts an inner intermediate representation (embedding) of the user vectors
    from a saved autoencoder model
"""

# Copyright (c) 2017 NVIDIA Corporation
import argparse
import copy
import json
from pathlib import Path

import torch
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
    default="out.txt",
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
    # Loads training data
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

    # Initializes model in "deep feature extraction" mode
    rencoder = model.AutoEncoder(
        layer_sizes=[data_layer.vector_dim]
        + [int(l_sizes) for l_sizes in args.hidden_layers.split(",")],
        nl_type=args.non_linearity_type,
        is_constrained=args.constrained,
        dp_drop_prob=args.drop_prob,
        last_layer_activations=not args.skip_last_layer_nl,
        extract_deep_features=True,
    )

    # Loads pre-trained model
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
    eval_data_layer.src_data = data_layer.data

    user_deepfeatures_dict = dict()

    # Extracts intermediate representation for each user
    for i, ((_, src), majorInd) in enumerate(
        eval_data_layer.iterate_one_epoch_eval(for_inf=True)
    ):
        # input for the autoencoder
        user_ratings = Variable(src.cuda().to_dense() if use_gpu else src.to_dense())

        # rencoder is built to output the deep features
        deep_features = rencoder(user_ratings).cpu().data.numpy()[0, :]

        # get user idx as in the train data
        user_idx = inv_userIdMap[majorInd]

        if user_idx not in user_deepfeatures_dict:
            # new user: save deep features in the dictionary
            user_deepfeatures_dict[user_idx] = []
            for ind in range(32):  # 32 deep features
                user_deepfeatures_dict[user_idx].append(str(deep_features[ind]))

            if i % 1000 == 0:
                print("Extracted deep features for {} users".format(i))

    print("Saving deep feature dictionary..")
    with open(args.predictions_path, "w") as outf:
        json.dump(user_deepfeatures_dict, outf)

    print("Saved deep feature dictionary to:", args.predictions_path)


if __name__ == "__main__":
    main()
