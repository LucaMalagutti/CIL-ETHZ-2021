import argparse
import os
import random
from math import sqrt

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import RatingDataset
from reco_encoder.model import model

# from run import set_optimizer
from torch import nn, optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser(description="Alternated MLP+DeepRec")
parser.add_argument(
    "--pretrain_autoencoders",
    action="store_true",
    help="Pretrain the autoencoders before starting mlp+autoencoder training",
)
parser.add_argument(
    "--train_mlp_every",
    type=int,
    default=1,
    help="train mlp after encoder on one every X batches",
)
parser.add_argument(
    "--eval_every",
    type=int,
    default=5,
    help="perform evaluation on one every X epochs",
)
parser.add_argument(
    "--learning_rate", type=float, default=0.00001, metavar="N", help="learning rate"
)
parser.add_argument(
    "--dropout", type=float, default=0.0, help="dropout drop probability"
)
parser.add_argument(
    "--num_epochs", type=int, default=50, help="maximum number of epochs"
)
parser.add_argument(
    "--pretrain_num_epochs",
    type=int,
    default=3,
    help="maximum number of epochs",
)
parser.add_argument(
    "--save_every",
    type=int,
    default=1,
    help="save every N number of epochs",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="momentum",
    help="optimizer kind: adam, momentum, adagrad or rmsprop",
)
parser.add_argument(
    "--layer_sizes",
    type=str,
    default="64,32,16",
    help="hidden layer sizes",
)
parser.add_argument(
    "--gpu_ids",
    type=str,
    default="0",
    help="comma-separated gpu ids to use for data parallel training",
)
parser.add_argument(
    "--path_to_user_embs",
    type=str,
    default="data/embs/users_emb.pckl",
    help="Path to user emb",
)
parser.add_argument(
    "--path_to_item_embs",
    type=str,
    default="data/embs/items_emb.pckl",
    help="Path to item emb",
)
parser.add_argument(
    "--path_to_train_data",
    type=str,
    default="data/train90/CIL_data90.train",
    help="Path to training data",
)
parser.add_argument(
    "--path_to_eval_data",
    type=str,
    default="data/valid/CIL_data10.valid",
    help="Path to evaluation data",
)
parser.add_argument(
    "--logdir",
    type=str,
    default="model_save/alternate/",
    help="where to save model and write logs",
)

args = parser.parse_args()
args.layer_sizes = [int(x) for x in args.layer_sizes.split(",")]


def set_optimizer(optimizer, lr, weight_decay, rencoder):
    optimizers = {
        "adam": optim.Adam(rencoder.parameters(), lr=lr, weight_decay=weight_decay),
        "adagrad": optim.Adagrad(
            rencoder.parameters(), lr=lr, weight_decay=weight_decay
        ),
        "momentum": optim.SGD(
            rencoder.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        ),
        "rmsprop": optim.RMSprop(
            rencoder.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        ),
    }

    try:
        return optimizers[optimizer]
    except ValueError:
        raise ValueError("Unknown optimizer kind")


def main():
    wandb.init(
        project="CIL-2021",
        entity="spaghetticode",
        config={
            "mlp_learning_rate": args.learning_rate,
            # TODO add all hyperparameters
        },
    )

    if not os.path.exists("model_save/"):
        os.mkdir("model_save/")
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    layer_sizes_items = [10000, 2048, 512, 512]
    layer_sizes_users = [1000, 512, 32, 64]
    dp_items = 0.4
    dp_users = 0.4
    learning_rate_items = 0.01
    weight_decay_items = 0.001
    learning_rate_users = 0.0035
    weight_decay_users = 5.0e-08
    encoder_batch_size = 64
    dense_refeeding_steps = 2

    model_checkpoint_items = (
        f"model_save/pretrain_emb/items/model.epoch_{args.pretrain_num_epochs}"
    )

    items_train_cmd = f"""python3 run.py --num_epochs={args.pretrain_num_epochs}
                                        --batch_size={encoder_batch_size}
                                        --dense_refeeding_steps={dense_refeeding_steps}
                                        --dropout={dp_items}
                                        --layer1_dim={layer_sizes_items[1]}
                                        --layer2_dim={layer_sizes_items[2]}
                                        --layer3_dim={layer_sizes_items[3]}
                                        --learning_rate={learning_rate_items}
                                        --weight_decay={weight_decay_items}
                                        --major=items
                                        --save_every={args.pretrain_num_epochs}
                                        --logdir=model_save/pretrain_emb/items """

    items_train_cmd = (
        items_train_cmd.replace("\n", "").replace("\t", "").replace("\r", "")
    )

    model_checkpoint_users = (
        f"model_save/pretrain_emb/users/model.epoch_{args.pretrain_num_epochs}"
    )

    users_train_cmd = f"""python3 run.py --num_epochs={args.pretrain_num_epochs}
                                --batch_size={encoder_batch_size}
                                --dense_refeeding_steps={dense_refeeding_steps}
                                --dropout={dp_users}
                                --layer1_dim={layer_sizes_users[1]}
                                --layer2_dim={layer_sizes_users[2]}
                                --layer3_dim={layer_sizes_users[3]}
                                --learning_rate={learning_rate_users}
                                --weight_decay={weight_decay_users}
                                --save_every={args.pretrain_num_epochs}
                                --major=users
                                --logdir=model_save/pretrain_emb/users"""

    users_train_cmd = (
        users_train_cmd.replace("\n", "").replace("\t", "").replace("\r", "")
    )

    if args.pretrain_autoencoders:
        os.system(users_train_cmd)
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

    if args.pretrain_autoencoders:
        print("Loading model from: {}".format(model_checkpoint_items))
        items_encoder.load_state_dict(torch.load(model_checkpoint_items))

        print("Loading model from: {}".format(model_checkpoint_users))
        users_encoder.load_state_dict(torch.load(model_checkpoint_users))

    mlp = model.MLP(
        layer_sizes_items[-1] + layer_sizes_users[-1], args.layer_sizes, args.dropout
    )

    print(users_encoder)
    print(items_encoder)
    print(mlp)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        users_encoder = users_encoder.cuda()
        items_encoder = items_encoder.cuda()
        mlp = mlp.cuda()

    items_optimizer = set_optimizer(
        args.optimizer,
        learning_rate_items,
        weight_decay_items,
        items_encoder,
    )
    items_scheduler = MultiStepLR(
        items_optimizer, milestones=[24, 36, 48, 66, 72], gamma=0.5
    )

    users_optimizer = set_optimizer(
        args.optimizer,
        learning_rate_users,
        weight_decay_users,
        users_encoder,
    )
    users_scheduler = MultiStepLR(
        users_optimizer, milestones=[24, 36, 48, 66, 72], gamma=0.5
    )

    mlp_optimizer = optim.Adam(mlp.parameters(), lr=args.learning_rate)
    mlp_criterion = model.RMSELoss()

    train_dataset = RatingDataset(args.path_to_train_data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=encoder_batch_size,
        shuffle=True,
        drop_last=True,
    )

    eval_dataset = RatingDataset(args.path_to_eval_data)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=encoder_batch_size,
        drop_last=True,
    )

    for epoch_i in range(1, args.num_epochs + 1):
        user_tot_epoch_loss = 0
        item_tot_epoch_loss = 0
        user_denom = 0
        item_denom = 0
        mlp_tot_epoch_loss = 0

        print(f"EPOCH: {epoch_i}")
        for batch_i, batch in enumerate(tqdm(train_dataloader)):
            users_encoder.train()
            items_encoder.train()

            user_vectors, item_vectors, ratings = batch

            user_vectors = user_vectors.cuda() if use_gpu else user_vectors
            item_vectors = item_vectors.cuda() if use_gpu else item_vectors
            ratings = ratings.cuda() if use_gpu else ratings

            users_optimizer.zero_grad()
            items_optimizer.zero_grad()

            user_outputs = users_encoder(user_vectors.float())
            item_outputs = items_encoder(item_vectors.float())

            user_loss, user_num_ratings = model.MSEloss(
                user_outputs, user_vectors.float()
            )
            item_loss, item_num_ratings = model.MSEloss(
                item_outputs, item_vectors.float()
            )
            user_loss /= user_num_ratings
            item_loss /= item_num_ratings

            user_loss.backward()
            item_loss.backward()

            users_optimizer.step()
            items_optimizer.step()

            user_tot_epoch_loss += user_loss.item()
            item_tot_epoch_loss += item_loss.item()
            user_denom += 1
            item_denom += 1

            if dense_refeeding_steps > 0:
                for _ in range(dense_refeeding_steps):
                    user_inputs = Variable(user_outputs.data)
                    item_inputs = Variable(item_outputs.data)

                    users_optimizer.zero_grad()
                    items_optimizer.zero_grad()

                    user_outputs = users_encoder(user_inputs)
                    item_outputs = items_encoder(item_inputs)

                    user_loss, user_num_ratings = model.MSEloss(
                        user_outputs, user_inputs
                    )
                    user_loss = user_loss / user_num_ratings
                    user_loss.backward()
                    users_optimizer.step()

                    item_loss, item_num_ratings = model.MSEloss(
                        item_outputs, item_inputs
                    )
                    item_loss = item_loss / item_num_ratings
                    item_loss.backward()
                    items_optimizer.step()

            users_scheduler.step()
            items_scheduler.step()

            if batch_i % args.train_mlp_every == 0:
                mlp.train()
                mlp_optimizer.zero_grad()

                users_encoder.eval()
                items_encoder.eval()

                user_embeddings = users_encoder.extract_embeddings(user_inputs)
                item_embeddings = items_encoder.extract_embeddings(item_inputs)

                mlp_inputs = torch.cat((user_embeddings, item_embeddings), 1)
                mlp_inputs = mlp_inputs.cuda() if use_gpu else mlp_inputs

                # print(mlp_inputs.shape)
                mlp_outputs = mlp(mlp_inputs.float())

                mlp_loss = mlp_criterion(mlp_outputs, ratings.float().view(-1, 1))
                mlp_loss.backward()
                mlp_optimizer.step()

                mlp_tot_epoch_loss += mlp_loss.item()

        train_user_loss = sqrt(user_tot_epoch_loss / user_denom)
        train_item_loss = sqrt(item_tot_epoch_loss / item_denom)
        train_mlp_loss = mlp_tot_epoch_loss / len(train_dataloader)
        print(
            f"""Epoch {epoch_i} TRAINING RMSE
             \n\t user_loss: {train_user_loss}
             \n\t item_loss: {train_item_loss}
             \n\t mlp_loss: {train_mlp_loss}
             """
        )
        wandb.log(
            {
                "train_user_loss": train_user_loss,
                "train_item_loss": train_item_loss,
                "train_mlp_loss": train_mlp_loss,
            },
            step=epoch_i,
        )

        if epoch_i & args.eval_every == 0:
            users_encoder.eval()
            items_encoder.eval()
            mlp.eval()

            eval_user_tot_epoch_loss = 0
            eval_item_tot_epoch_loss = 0
            eval_user_denom = 0
            eval_item_denom = 0
            eval_mlp_tot_epoch_loss = 0

            for batch_i, batch in enumerate(tqdm(eval_dataloader)):
                user_vectors, item_vectors, ratings = batch

                user_vectors = user_vectors.cuda() if use_gpu else user_vectors
                item_vectors = item_vectors.cuda() if use_gpu else item_vectors

                user_outputs = users_encoder(user_vectors.float())
                item_outputs = items_encoder(item_vectors.float())

                eval_user_loss, user_num_ratings = model.MSEloss(
                    user_outputs, user_vectors.float()
                )
                eval_item_loss, item_num_ratings = model.MSEloss(
                    item_outputs, item_vectors.float()
                )
                eval_user_loss /= user_num_ratings
                eval_item_loss /= item_num_ratings

                eval_user_tot_epoch_loss += eval_user_loss.item()
                eval_item_tot_epoch_loss += eval_item_loss.item()
                eval_user_denom += 1
                eval_item_denom += 1

                user_embeddings = users_encoder.extract_embeddings(user_inputs)
                item_embeddings = items_encoder.extract_embeddings(item_inputs)

                mlp_inputs = torch.cat((user_embeddings, item_embeddings), 1)
                mlp_inputs = mlp_inputs.cuda() if use_gpu else mlp_inputs

                # print(mlp_inputs.shape)
                mlp_outputs = mlp(mlp_inputs.float())

                eval_mlp_loss = mlp_criterion(mlp_outputs, ratings.float().view(-1, 1))

                eval_mlp_tot_epoch_loss += eval_mlp_loss.item()

            eval_user_loss = sqrt(eval_user_tot_epoch_loss / eval_user_denom)
            eval_item_loss = sqrt(eval_item_tot_epoch_loss / eval_item_denom)
            eval_mlp_loss = eval_mlp_tot_epoch_loss / len(eval_dataloader)
            print(
                f"""Epoch {epoch_i} EVALUATION RMSE
                \n\t user_loss: {eval_user_loss}
                \n\t item_loss: {eval_item_loss}
                \n\t mlp_loss: {eval_mlp_loss}
                """
            )
            wandb.log(
                {
                    "eval_user_loss": eval_user_loss,
                    "eval_item_loss": eval_item_loss,
                    "eval_mlp_loss": eval_mlp_loss,
                },
                step=epoch_i,
            )

            torch.save(mlp.state_dict(), args.logdir + "mlp_epoch_" + str(epoch_i))
            torch.save(
                users_encoder.state_dict(),
                args.logdir + "users_encoder_epoch_" + str(epoch_i),
            )
            torch.save(
                items_encoder.state_dict(),
                args.logdir + "items_encoder_epoch_" + str(epoch_i),
            )


if __name__ == "__main__":
    main()
