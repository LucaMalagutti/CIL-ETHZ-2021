import argparse
import os
import pickle
import random

import numpy as np
import pandas
import run
import torch
import wandb
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
    "--pretrain_num_epochs",
    type=int,
    default=10,
    metavar="N",
    help="maximum number of epochs",
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

layer_sizes_items = [10000, 2048, 512, 512]
layer_sizes_users = [1000, 512, 32, 64]
dp_items = 0.4
dp_users = 0.4


model_checkpoint_items = (
    f"model_save/pretrain_emb/items/model.epoch_{args.pretrain_num_epochs}"
)

items_train_cmd = f"""python3 run.py --num_epochs={args.pretrain_num_epochs}
                                    --batch_size=128
                                    --dense_refeeding_steps=2
                                    --dropout={dp_items}
                                    --layer1_dim={layer_sizes_items[1]}
                                    --layer2_dim={layer_sizes_items[2]}
                                    --layer3_dim={layer_sizes_items[3]}
                                    --learning_rate=0.01
                                    --major=items
                                    --weight_decay=0.001
                                    --save_every={args.pretrain_num_epochs}
                                    --logdir=model_save/pretrain_emb/items """

items_train_cmd = items_train_cmd.replace("\n", "").replace("\t", "").replace("\r", "")

os.system(items_train_cmd)

model_checkpoint_users = (
    f"model_save/pretrain_emb/users/model.epoch_{args.pretrain_num_epochs}"
)

users_train_cmd = f"""python3 run.py --num_epochs={args.pretrain_num_epochs}
                            --batch_size=8
                            --dense_refeeding_steps=3
                            --dropout={dp_users}
                            --layer1_dim={layer_sizes_users[1]}
                            --layer2_dim={layer_sizes_users[2]}
                            --layer3_dim={layer_sizes_users[3]}
                            --learning_rate=0.0035
                            --weight_decay=5.0e-08
                            --save_every={args.pretrain_num_epochs}
                            --major=users
                            --logdir=model_save/pretrain_emb/users"""

users_train_cmd = users_train_cmd.replace("\n", "").replace("\t", "").replace("\r", "")

os.system(items_train_cmd)

items_encoder = model.AutoEncoder(
    layer_sizes=layer_sizes_items,
    is_constrained=False,
    dp_drop_prob=dp_items,
)

users_encoder = model.AutoEncoder(
    layer_sizes=layer_sizes_users,
    is_constrained=False,
    dp_drop_prob=dp_users,
)


print("Loading model from: {}".format(model_checkpoint_items))
items_encoder.load_state_dict(torch.load(model_checkpoint_items))

print("Loading model from: {}".format(model_checkpoint_users))
users_encoder.load_state_dict(torch.load(model_checkpoint_users))


print(users_encoder)
print(items_encoder)
