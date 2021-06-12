"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
"""
import multiprocessing
from pprint import pprint
from time import time

import dataloader
import model
import numpy as np
import torch
import utils
import world
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils import timer

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, model, loss_class, epoch, w=None):
    model.train()
    bpr: utils.BPRLoss = loss_class

    with timer(name="Epoch Time"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    pos_items = torch.Tensor(S[:, 1]).long()
    neg_items = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    pos_items = pos_items.to(world.device)
    neg_items = neg_items.to(world.device)
    users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)
    total_batch = len(users) // world.config["bpr_batch_size"] + 1
    aver_loss = 0.0
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
        utils.minibatch(
            users, pos_items, neg_items, batch_size=world.config["bpr_batch_size"]
        )
    ):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(
                "BPRLoss/BPR",
                cri,
                epoch * int(len(users) / world.config["bpr_batch_size"]) + batch_i,
            )
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss {aver_loss:.3f}-{time_info}"


def test_one_batch(X):
    sorted_items = X[0].numpy()
    ground_truth = X[1]
    r = utils.getLabel(ground_truth, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(ground_truth, r, k)
        pre.append(ret["precision"])
        recall.append(ret["recall"])
        ndcg.append(utils.NDCGatK_r(ground_truth, r, k))
    return {
        "recall": np.array(recall),
        "precision": np.array(pre),
        "ndcg": np.array(ndcg),
    }


def test(dataset, model, epoch, w=None, multicore=False):
    u_batch_size = world.config["test_u_batch_size"]
    dataset: utils.BasicDataset
    test_dict: dict = dataset.get_test_dict
    model: model.LightGCN
    # eval mode with no dropout
    model = model.eval()
    max_K = max(world.topks)
    if multicore is True:
        pool = multiprocessing.Pool(CORES)
    results = {
        "precision": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "ndcg": np.zeros(len(world.topks)),
    }
    with torch.no_grad():
        users = list(test_dict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(
                f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}"
            )
        users_list = []
        rating_list = []
        ground_truth_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            all_pos = dataset.get_user_pos_items(batch_users)  # training samples
            ground_truth = [test_dict[u] for u in batch_users]  # testing samples
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = model.get_user_rating(batch_users_gpu)
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(all_pos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(ground_truth)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            ground_truth_list.append(ground_truth)
        assert total_batch == len(users_list)
        X = zip(rating_list, ground_truth_list)
        if multicore is True:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))

        for result in pre_results:
            results["recall"] += result["recall"]
            results["precision"] += result["precision"]
            results["ndcg"] += result["ndcg"]
        results["recall"] /= float(len(users))
        results["precision"] /= float(len(users))
        results["ndcg"] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(
                f"Test/Recall@{world.topks}",
                {
                    str(world.topks[i]): results["recall"][i]
                    for i in range(len(world.topks))
                },
                epoch,
            )
            w.add_scalars(
                f"Test/Precision@{world.topks}",
                {
                    str(world.topks[i]): results["precision"][i]
                    for i in range(len(world.topks))
                },
                epoch,
            )
            w.add_scalars(
                f"Test/NDCG@{world.topks}",
                {
                    str(world.topks[i]): results["ndcg"][i]
                    for i in range(len(world.topks))
                },
                epoch,
            )
        if multicore is True:
            pool.close()
        print(results)
        return results
