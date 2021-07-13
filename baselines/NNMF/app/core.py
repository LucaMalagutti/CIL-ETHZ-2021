import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import pandas as pd
from tqdm import tqdm

from .models import NNMF
from .utils import dataset
from .config import *
from . import utils


def _get_batch(train_data, batch_size):
    if batch_size:
        return train_data.sample(batch_size)
    return train_data


def _train(model, sess, saver, train_data, valid_data, batch_size):
    prev_valid_rmse = float("Inf")
    early_stop_iters = 0
    print(MAX_ITER)
    for i in range(MAX_ITER):
        # Run SGD
        batch = train_data.sample(batch_size) if batch_size else train_data
        model.train_iteration(batch)

        # Evaluate
        train_loss = model.eval_loss(batch)
        train_rmse = model.eval_rmse(batch)
        valid_loss = model.eval_loss(valid_data)
        valid_rmse = model.eval_rmse(valid_data)
        #print("{:3f} {:3f}, {:3f} {:3f}".format(train_loss, train_rmse,
        #                                        valid_loss, valid_rmse))

        if EARLY_STOP:
            early_stop_iters += 1
            if valid_rmse < prev_valid_rmse:
                prev_valid_rmse = valid_rmse
                early_stop_iters = 0
                saver.save(sess, model.model_file_path)
            elif early_stop_iters >= EARLY_STOP_MAX_ITER:
                print("Early stopping ({} vs. {})...".format(
                    prev_valid_rmse, valid_rmse))
                break
        else:
            saver.save(sess, model.model_file_path)


def _test(model, valid_data, test_data):
    valid_rmse = model.eval_rmse(valid_data)
    test_rmse = model.eval_rmse(test_data)
    print("Final valid RMSE: {}, test RMSE: {}".format(valid_rmse, test_rmse))
    return valid_rmse, test_rmse


def run(batch_size=None, **hyper_params):
    # kind = dataset.ML_100K
    # kind = dataset.ML_1M
    kind = dataset.NETFLIX

    with tf.Session() as sess:
        # Process data
        print("Reading in data")
        data = dataset.load_data(kind)

        # Define computation graph & Initialize
        print('Building network & initializing variables')
        model = NNMF(kind, **hyper_params)
        model.init_sess(sess)
        saver = tf.train.Saver()

        _train(
            model,
            sess,
            saver,
            data['train'],
            data['valid'],
            batch_size=batch_size)

        print('Loading best checkpointed model')
        saver.restore(sess, model.model_file_path)
        valid_rmse, test_rmse = _test(model, data['valid'], data['test'])

        _COL_NAMES = ['user_id', 'item_id', 'rating']
        _DELIMITER = ','
        sub_data = pd.read_csv("data/netflix/sub.csv",
            delimiter=_DELIMITER,
            header=0,
            names=_COL_NAMES)

        print(sub_data.head())

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
