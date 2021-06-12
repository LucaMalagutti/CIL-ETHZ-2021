"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
import sys
from os.path import join
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import world
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset
from world import cprint


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def get_n_users(self):
        raise NotImplementedError

    @property
    def get_m_items(self):
        raise NotImplementedError

    @property
    def get_train_data_size(self):
        raise NotImplementedError

    @property
    def get_test_dict(self):
        raise NotImplementedError

    @property
    def get_items_per_user(self):
        raise NotImplementedError

    def get_user_pos_items(self, users):
        raise NotImplementedError

    def get_user_neg_items(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def get_sparse_graph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |0,   R|
            |R.T, 0|
        """
        raise NotImplementedError

class CIL(BasicDataset):
    def __init__(self, config=world.config, path="../data/cil"):
        cprint("loading [CIL dataset]")
        self.split = config["A_split"]
        self.folds = config["A_n_fold"]
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]
        train_file = path + "/train.csv"
        test_file = path + "/test.csv"
        self.path = path

        # read adjacency matrix
        train_df = pd.read_csv(train_file)
        self.trainUniqueUsers = np.array(train_df['user'].unique())
        test_df = pd.read_csv(test_file)

        self.testUniqueUsers = np.array(test_df['user'].unique())
        self.train_data_size = train_df.shape[0]
        self.test_data_size = test_df.shape[0]

        self.user_item_net = sp.load_npz("../data/cil/adj_mat.npz")
        self.test_user_item_net = sp.load_npz("../data/cil/test_adj_mat.npz")
        self.n_user = self.user_item_net.shape[0]
        self.m_item = self.user_item_net.shape[1]
        self.Graph = None
        print(f"{self.train_data_size} interactions for training")
        print(f"{self.test_data_size} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.train_data_size + self.test_data_size) / self.n_user / self.m_item}")

        self.users_D = np.array(self.user_item_net.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.0] = 3.0  # TODO: temporary value
        self.items_D = np.array(self.user_item_net.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.0] = 3.0  # TODO: temporary value

        # pre-calculate
        self.items_per_user = self.get_user_pos_items(list(range(self.n_user)))
        self.test_dict = self.__build_test(self.testUniqueUsers)
        print(f"{world.dataset} is ready to go")

    @property
    def get_n_users(self):
        return self.n_user

    @property
    def get_m_items(self):
        return self.m_item

    @property
    def get_train_data_size(self):
        return self.train_data_size

    @property
    def get_test_dict(self):
        return self.test_dict

    @property
    def get_items_per_user(self):
        return self.items_per_user

    def get_user_pos_items(self, users):
        pos_items = []
        for user in users:
            pos_items.append(self.user_item_net[user].nonzero()[1])
        return pos_items

    def __build_test(self, users):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for user in users:
            test_data[user] = self.test_user_item_net[user].nonzero()[1]
        return test_data

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_sparse_graph(self):
        print("loading A matrix")
        if self.Graph is None:
            try:
                A_tilde = sp.load_npz(self.path + "/A_tilde.npz")
                print("...successfully loaded")
            except Exception:
                print("generating adjacency matrix...")
                s = time()
                A = sp.dok_matrix(
                    (self.n_user + self.m_item, self.n_user + self.m_item),
                    dtype=np.float32,
                )
                A = A.tolil()
                R = self.user_item_net.tolil()
                A[: self.n_user, self.n_user:] = R
                A[self.n_user:, : self.n_user] = R.T

                rowsum = np.array(A.sum(axis=1))
                D_inv = np.power(rowsum, -0.5).flatten()
                D_inv[np.isinf(D_inv)] = 0.0
                D = sp.diags(D_inv)

                A_tilde = D.dot(A).dot(D)
                A_tilde = A_tilde.tocsr()

                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + "/A_tilde.npz", A_tilde)

            self.Graph = self._convert_sp_mat_to_sp_tensor(A_tilde)
            self.Graph = self.Graph.coalesce().to(world.device)

        return self.Graph

class LastFM(BasicDataset):
    """
    Dataset type for pytorch
    include graph information
    LastFM dataset
    """

    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]
        trainData = pd.read_table(join(path, "data1.txt"), header=None)
        testData = pd.read_table(join(path, "test1.txt"), header=None)
        trustNet = pd.read_table(join(path, "trustnetwork.txt"), header=None).to_numpy()
        trustNet -= 1
        trainData -= 1
        testData -= 1
        self.trustNet = trustNet
        self.trainData = trainData
        self.testData = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        self.testUser = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem = np.array(testData[:][1])
        self.Graph = None
        self.n_user = 1892
        self.m_item = 4489
        print(
            f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_user/self.m_item}"
        )

        # (users,users)
        self.socialNet = csr_matrix(
            (np.ones(len(trustNet)), (trustNet[:, 0], trustNet[:, 1])),
            shape=(self.n_user, self.n_user),
        )
        # (users,items), bipartite graph
        self.user_item_net = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item),
        )

        # pre-calculate
        self.items_per_user = self.get_user_pos_items(list(range(self.n_user)))
        self.allNeg = []
        allItems = set(range(self.m_item))
        for i in range(self.n_user):
            pos = set(self.items_per_user[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.test_dict = self.__build_test()

    @property
    def get_n_users(self):
        return self.n_user

    @property
    def get_m_items(self):
        return self.m_item

    @property
    def get_train_data_size(self):
        return len(self.trainUser)

    @property
    def get_test_dict(self):
        return self.test_dict

    @property
    def get_items_per_user(self):
        return self.items_per_user

    def get_sparse_graph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_user])
            second_sub = torch.stack([item_dim + self.n_user, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(
                index,
                data,
                torch.Size([self.n_user + self.m_item, self.n_user + self.m_item]),
            )
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D == 0.0] = 1.0
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense / D_sqrt
            dense = dense / D_sqrt.t()
            index = dense.nonzero()
            data = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(
                index.t(),
                data,
                torch.Size([self.n_user + self.m_item, self.n_user + self.m_item]),
            )
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def get_user_pos_items(self, users):
        pos_items = []
        for user in users:
            pos_items.append(self.user_item_net[user].nonzero()[1])
        return pos_items

    def get_user_neg_items(self, users):
        neg_items = []
        for user in users:
            neg_items.append(self.allNeg[user])
        return neg_items

    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict["test"]

    def __len__(self):
        return len(self.trainUniqueUsers)


class Loader(BasicDataset):
    """
    Dataset type for pytorch
    include graph information
    gowalla dataset
    """

    def __init__(self, config=world.config, path="../data/gowalla"):
        # train or test
        cprint(f"loading [{path}]")
        self.split = config["A_split"]
        self.folds = config["A_n_fold"]
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]
        self.n_user = 0
        self.m_item = 0
        train_file = path + "/train.txt"
        test_file = path + "/test.txt"
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.train_data_size = 0
        self.test_data_size = 0

        with open(train_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip("\n").split(" ")
                    items = [int(i) for i in line[1:]]
                    uid = int(line[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.train_data_size += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip("\n").split(" ")
                    items = [int(i) for i in line[1:]]
                    uid = int(line[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.test_data_size += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.Graph = None
        print(f"{self.train_data_size} interactions for training")
        print(f"{self.test_data_size} interactions for testing")
        print(
            f"{world.dataset} Sparsity : {(self.train_data_size + self.test_data_size) / self.n_user / self.m_item}"
        )

        # (users,items), bipartite graph
        self.user_item_net = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item),
        )
        self.users_D = np.array(self.user_item_net.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.0] = 1
        self.items_D = np.array(self.user_item_net.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.0] = 1.0
        # pre-calculate
        self.items_per_user = self.get_user_pos_items(list(range(self.n_user)))
        self.test_dict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def get_n_users(self):
        return self.n_user

    @property
    def get_m_items(self):
        return self.m_item

    @property
    def get_train_data_size(self):
        return self.train_data_size

    @property
    def get_test_dict(self):
        return self.test_dict

    @property
    def get_items_per_user(self):
        return self.items_per_user

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_user + self.m_item) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_user + self.m_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(
                self._convert_sp_mat_to_sp_tensor(A[start:end])
                .coalesce()
                .to(world.device)
            )
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_sparse_graph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + "/s_pre_adj_mat.npz")
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except Exception:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix(
                    (self.n_user + self.m_item, self.n_user + self.m_item),
                    dtype=np.float32,
                )
                adj_mat = adj_mat.tolil()
                R = self.user_item_net.tolil()
                adj_mat[: self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, : self.n_user] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.0
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + "/s_pre_adj_mat.npz", norm_adj)

            if self.split is True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def get_user_pos_items(self, users):
        pos_items = []
        for user in users:
            pos_items.append(self.user_item_net[user].nonzero()[1])
        return pos_items
