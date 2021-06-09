# Copyright (c) 2017 NVIDIA Corporation
import argparse
import copy
import os
import time
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb as wandb
from logger import Logger
from reco_encoder.data import input_layer
from reco_encoder.model import model
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser(description="RecoEncoder")
parser.add_argument(
    "--lr", type=float, default=0.00001, metavar="N", help="learning rate"
)
parser.add_argument(
    "--weight_decay", type=float, default=0.0, metavar="N", help="L2 weight decay"
)
parser.add_argument(
    "--drop_prob", type=float, default=0.0, metavar="N", help="dropout drop probability"
)
parser.add_argument(
    "--noise_prob", type=float, default=0.0, metavar="N", help="noise probability"
)
parser.add_argument(
    "--batch_size", type=int, default=64, metavar="N", help="global batch size"
)
parser.add_argument(
    "--summary_frequency",
    type=int,
    default=100,
    metavar="N",
    help="how often to save summaries",
)
parser.add_argument(
    "--aug_step",
    type=int,
    default=-1,
    metavar="N",
    help="do data augmentation every X step",
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
    "--num_epochs", type=int, default=50, metavar="N", help="maximum number of epochs"
)
parser.add_argument(
    "--save_every",
    type=int,
    default=3,
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
    "--hidden_layers",
    type=str,
    default="1024,512,512,128",
    metavar="N",
    help="hidden layer sizes, comma-separated",
)
parser.add_argument(
    "--gpu_ids",
    type=str,
    default="0",
    metavar="N",
    help="comma-separated gpu ids to use for data parallel training",
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
    "--logdir",
    type=str,
    default="logs",
    metavar="N",
    help="where to save model and write logs",
)

args = parser.parse_args()
print(args)

use_gpu = torch.cuda.is_available()  # global flag
if use_gpu:
    print("GPU is available.")
else:
    print("GPU is not available.")


def do_eval(encoder, evaluation_data_layer):
    encoder.eval()
    denom = 0.0
    total_epoch_loss = 0.0
    for i, (eval, src) in enumerate(evaluation_data_layer.iterate_one_epoch_eval()):
        inputs = Variable(src.cuda().to_dense() if use_gpu else src.to_dense())
        targets = Variable(eval.cuda().to_dense() if use_gpu else eval.to_dense())
        outputs = encoder(inputs)
        loss, num_ratings = model.MSEloss(outputs, targets)
        total_epoch_loss += loss.item()
        denom += num_ratings.item()
    return sqrt(total_epoch_loss / denom)


def log_var_and_grad_summaries(
    logger, layers, global_step, prefix, log_histograms=False
):
    """
    Logs variable and grad stats for layer. Transfers data from GPU to CPU automatically
    :param logger: TB logger
    :param layers: param list
    :param global_step: global step for TB
    :param prefix: name prefix
    :param log_histograms: (default: False) whether or not log histograms
    :return:
    """
    for ind, w in enumerate(layers):
        # Variables
        w_var = w.data.cpu().numpy()
        logger.scalar_summary(
            "Variables/FrobNorm/{}_{}".format(prefix, ind),
            np.linalg.norm(w_var),
            global_step,
        )
        if log_histograms:
            logger.histo_summary(
                tag="Variables/{}_{}".format(prefix, ind),
                values=w.data.cpu().numpy(),
                step=global_step,
            )

        # Gradients
        w_grad = w.grad.data.cpu().numpy()
        logger.scalar_summary(
            "Gradients/FrobNorm/{}_{}".format(prefix, ind),
            np.linalg.norm(w_grad),
            global_step,
        )
        if log_histograms:
            logger.histo_summary(
                tag="Gradients/{}_{}".format(prefix, ind),
                values=w.grad.data.cpu().numpy(),
                step=global_step,
            )


def set_optimizer(args, rencoder):
    optimizers = {
        "adam": optim.Adam(
            rencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
        ),
        "adagrad": optim.Adagrad(
            rencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
        ),
        "momentum": optim.SGD(
            rencoder.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        ),
        "rmsprop": optim.RMSprop(
            rencoder.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        ),
    }

    try:
        return optimizers[args.optimizer]
    except ValueError:
        raise ValueError("Unknown optimizer kind")


def main():
    logger = Logger(args.logdir)
    # 1. Start a new run
    wandb.init(
        project="CIL-2021",
        entity="spaghetticode",
        config={
            "batch_size": args.batch_size,
            "layer1_dim": args.hidden_layers.split(",")[0],
            "layer2_dim": args.hidden_layers.split(",")[1],
            "layer3_dim": args.hidden_layers.split(",")[2],
            "activation": args.non_linearity_type,
            "optimizer": args.optimizer,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "noise_prob": args.noise_prob,
            "dropout": args.drop_prob,
            "dense_refeeding_steps": args.aug_step,
        },
    )

    params = dict()
    params["batch_size"] = args.batch_size
    params["data_dir"] = args.path_to_train_data
    params["major"] = "users"
    params["itemIdInd"] = 1
    params["userIdInd"] = 0
    print("Loading training data")
    data_layer = input_layer.UserItemRecDataProvider(params=params)
    print("Data loaded")
    print("Total items found: {}".format(len(data_layer.data.keys())))
    print("Vector dim: {}".format(data_layer.vector_dim))

    if args.path_to_eval_data != "":
        print("Loading validation data")
        eval_params = copy.deepcopy(params)
        # must set eval batch size to 1 to make sure no examples are missed
        eval_params["data_dir"] = args.path_to_eval_data
        eval_data_layer = input_layer.UserItemRecDataProvider(
            params=eval_params,
            user_id_map=data_layer.userIdMap,  # the mappings are provided
            item_id_map=data_layer.itemIdMap,
        )
        eval_data_layer.src_data = data_layer.data
        print(
            "Total validation items found: {}".format(len(eval_data_layer.data.keys()))
        )
        print("Vector dim: {}".format(eval_data_layer.vector_dim))
    else:
        print("Skipping eval data")

    rencoder = model.AutoEncoder(
        layer_sizes=[data_layer.vector_dim]
        + [int(l_sizes) for l_sizes in args.hidden_layers.split(",")],
        nl_type=args.non_linearity_type,
        is_constrained=args.constrained,
        dp_drop_prob=args.drop_prob,
        last_layer_activations=not args.skip_last_layer_nl,
    )
    wandb.watch(rencoder)
    os.makedirs(args.logdir, exist_ok=True)
    model_checkpoint = args.logdir + "/model"
    path_to_model = Path(model_checkpoint)
    if path_to_model.is_file():
        print("Loading model from: {}".format(model_checkpoint))
        rencoder.load_state_dict(torch.load(model_checkpoint))

    print("######################################################")
    print("######################################################")
    print("############# AutoEncoder Model: #####################")
    print(rencoder)
    print("######################################################")
    print("######################################################")

    gpu_ids = [int(g) for g in args.gpu_ids.split(",")]
    print("Using GPUs: {}".format(gpu_ids))
    if len(gpu_ids) > 1:
        rencoder = nn.DataParallel(rencoder, device_ids=gpu_ids)

    if use_gpu:
        rencoder = rencoder.cuda()

    optimizer = set_optimizer(args, rencoder)
    if args.optimizer == "momentum":
        scheduler = MultiStepLR(optimizer, milestones=[24, 36, 48, 66, 72], gamma=0.5)

    t_loss = 0.0
    t_loss_denom = 0.0
    global_step = 0

    if args.noise_prob > 0.0:
        dp = nn.Dropout(p=args.noise_prob)

    for epoch in range(args.num_epochs):
        print("Doing epoch {} of {}".format(epoch, args.num_epochs))
        e_start_time = time.time()
        rencoder.train()
        total_epoch_loss = 0.0
        denom = 0.0

        for i, mb in enumerate(data_layer.iterate_one_epoch()):
            inputs = Variable(mb.cuda().to_dense() if use_gpu else mb.to_dense())
            optimizer.zero_grad()
            outputs = rencoder(inputs)
            loss, num_ratings = model.MSEloss(outputs, inputs)
            loss = loss / num_ratings
            # wandb.log({"MSE_loss": loss})
            loss.backward()
            optimizer.step()
            global_step += 1
            t_loss += loss.item()
            t_loss_denom += 1

            if i % args.summary_frequency == 0:
                rmse = sqrt(t_loss / t_loss_denom)
                # wandb.log({"train_RMSE": rmse})
                print("[%d, %5d] RMSE: %.7f" % (epoch, i, rmse))
                logger.scalar_summary("Training_RMSE", rmse, global_step)
                t_loss = 0
                t_loss_denom = 0.0
                log_var_and_grad_summaries(
                    logger, rencoder.encode_w, global_step, "Encode_W"
                )
                log_var_and_grad_summaries(
                    logger, rencoder.encode_b, global_step, "Encode_b"
                )
                if not rencoder.is_constrained:
                    log_var_and_grad_summaries(
                        logger, rencoder.decode_w, global_step, "Decode_W"
                    )
                log_var_and_grad_summaries(
                    logger, rencoder.decode_b, global_step, "Decode_b"
                )

            total_epoch_loss += loss.item()
            denom += 1

            # if args.aug_step > 0 and i % args.aug_step == 0 and i > 0:
            if args.aug_step > 0:
                # Magic data augmentation trick happen here (dense refeeding)
                for t in range(args.aug_step):
                    inputs = Variable(outputs.data)
                    if args.noise_prob > 0.0:
                        inputs = dp(inputs)
                    optimizer.zero_grad()
                    outputs = rencoder(inputs)
                    loss, num_ratings = model.MSEloss(outputs, inputs)
                    loss = loss / num_ratings
                    # wandb.log({"MSE_loss": loss})
                    loss.backward()
                    optimizer.step()

        if args.optimizer == "momentum":
            scheduler.step()

        e_end_time = time.time()
        wandb.log({"train_RMSE": sqrt(total_epoch_loss / denom)})
        print(
            "Total epoch {} finished in {} seconds with TRAINING RMSE loss: {}".format(
                epoch, e_end_time - e_start_time, sqrt(total_epoch_loss / denom)
            )
        )
        logger.scalar_summary(
            "Training_RMSE_per_epoch", sqrt(total_epoch_loss / denom), epoch
        )
        logger.scalar_summary("Epoch_time", e_end_time - e_start_time, epoch)
        if epoch % args.save_every == 0 or epoch == args.num_epochs - 1:
            if args.path_to_eval_data != "":
                eval_loss = do_eval(rencoder, eval_data_layer)
                wandb.log({"val_RMSE": eval_loss, "epoch": epoch})
                print("Epoch {} EVALUATION LOSS: {}".format(epoch, eval_loss))
                logger.scalar_summary("EVALUATION_RMSE", eval_loss, epoch)
            else:
                print("Skipping evaluation")
            print(
                "Saving model to {}".format(model_checkpoint + ".epoch_" + str(epoch))
            )
            torch.save(rencoder.state_dict(), model_checkpoint + ".epoch_" + str(epoch))

    print("Saving model to {}".format(model_checkpoint + ".last"))
    torch.save(rencoder.state_dict(), model_checkpoint + ".last")

    print("Done")
    quit()

    # # save to onnx
    # dummy_input = Variable(
    #     torch.randn(params["batch_size"], data_layer.vector_dim).type(torch.float)
    # )
    # torch.onnx.export(
    #     rencoder.float(),
    #     dummy_input.cuda() if use_gpu else dummy_input,
    #     model_checkpoint + ".onnx",
    #     verbose=True,
    # )
    # print("ONNX model saved to {}!".format(model_checkpoint + ".onnx"))


if __name__ == "__main__":
    main()
