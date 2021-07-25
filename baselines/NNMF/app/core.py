"""Defines model train and test iterations"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import pandas as pd
from tqdm import tqdm

from .config import *
from .models import NNMF
from .utils import dataset


def _train(model, sess, saver, train_data, valid_data, batch_size):
    prev_valid_rmse = float("Inf")
    early_stop_iters = 0

    # prints training type and length variables
    print("Starting model training...")
    print(f"\t Max number of iterations: {MAX_ITER}, Early stopping: {EARLY_STOP}")

    for i in range(MAX_ITER):
        # Run SGD
        batch = train_data.sample(batch_size) if batch_size else train_data
        model.train_iteration(batch)

        # Evaluate
        train_loss = model.eval_loss(batch)
        train_rmse = model.eval_rmse(batch)
        valid_loss = model.eval_loss(valid_data)
        valid_rmse = model.eval_rmse(valid_data)

        # Prints training and evaluation results
        print(f"Iteration: {i}")
        print(f"\t Train loss: {train_loss:.1f}, Train RMSE: {train_rmse:.3f}")
        print(f"\t Val loss: {valid_loss:.1f}, Val RMSE: {valid_rmse:.3f}")

        # Stops training early if validation loss starts to increase
        if EARLY_STOP:
            early_stop_iters += 1
            if valid_rmse < prev_valid_rmse:
                prev_valid_rmse = valid_rmse
                early_stop_iters = 0
                # Saves the model checkpoint at this iteration
                saver.save(sess, model.model_file_path)
            elif early_stop_iters >= EARLY_STOP_MAX_ITER:
                print(
                    "Early stopping ({} vs. {})...".format(prev_valid_rmse, valid_rmse)
                )
                break
        else:
            # Saves the model checkpoint at this iteration
            saver.save(sess, model.model_file_path)


def _test(model, valid_data, test_data):
    # Evaluate model on iteration and test data
    # (run after training has finished)
    valid_rmse = model.eval_rmse(valid_data)
    test_rmse = model.eval_rmse(test_data)
    print("Final valid RMSE: {}, test RMSE: {}".format(valid_rmse, test_rmse))
    return valid_rmse, test_rmse


def run(batch_size=None, **hyper_params):
    kind = dataset.NETFLIX

    with tf.Session() as sess:
        # Process data
        print("Reading in data")
        data = dataset.load_data(kind)

        # Define and initialize model
        print("Building network & initializing variables")
        model = NNMF(kind, **hyper_params)
        model.init_sess(sess)
        saver = tf.train.Saver()

        # Starts training
        _train(model, sess, saver, data["train"], data["valid"], batch_size=batch_size)

        # Restores model with lowest validation RMSE
        print("Loading best model checkpoint")
        saver.restore(sess, model.model_file_path)
        valid_rmse, test_rmse = _test(model, data["valid"], data["test"])

        # Generates submission for Kaggle
        _COL_NAMES = ["user_id", "item_id", "rating"]
        _DELIMITER = ","
        sub_data = pd.read_csv(
            "data/netflix/sub.csv", delimiter=_DELIMITER, header=0, names=_COL_NAMES
        )

        sub_data_np = sub_data.to_numpy()

        scores_list = []
        for i in tqdm(range(sub_data_np.shape[0])):
            scores_list.append(model.predict(sub_data_np[i][0], sub_data_np[i][1]))

        sub_data["Prediction"] = scores_list
        sub_data["Id"] = (
            "r"
            + sub_data["user_id"].apply(lambda x: str(x + 1))
            + "_"
            + "c"
            + sub_data["item_id"].apply(lambda x: str(x + 1))
        )
        sub_data = sub_data[["Id", "Prediction"]]
        sub_data.to_csv("NNMF_sub.csv", index=False)

        return valid_rmse, test_rmse
