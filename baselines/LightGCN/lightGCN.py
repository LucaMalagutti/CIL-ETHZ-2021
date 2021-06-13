import os

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class LightGCN(nn.Module):
    def __init__(self, args):
        super(LightGCN, self).__init__()
        self.args = args

        self.n_users = 10000
        self.n_items = 1000

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=args.emb_size
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=args.emb_size
        )

        if args.restore_ckpt is None:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.A_tilde = self.build_A_tilde()

    def build_A_tilde(self):
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32,
        )
        A = A.tolil()
        R = sp.load_npz("data/R_train.npz")
        A[: self.n_users, self.n_users :] = R
        A[self.n_users :, : self.n_users] = R.T

        A_mask = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32,
        )
        A_mask = A_mask.tolil()
        R_mask = sp.load_npz("data/R_mask_train.npz")
        A_mask[: self.n_users, self.n_users :] = R_mask
        A_mask[self.n_users :, : self.n_users] = R_mask.T

        rowsum = np.array(A_mask.sum(axis=1))
        D_inv = np.power(rowsum, -0.5).flatten()
        D_inv[np.isinf(D_inv)] = 0.0
        D = sp.diags(D_inv)

        A_tilde = D.dot(A).dot(D)
        A_tilde = A_tilde.tocsr()

        A_tilde = self._convert_sp_mat_to_sp_tensor(A_tilde)

        return A_tilde.to(device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def forward(self, batch):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        curr_emb = torch.cat([users_emb, items_emb])
        embs_list = [curr_emb]

        # TODO: Add graph dropout

        for _ in range(self.args.n_layers):
            curr_emb = torch.sparse.mm(self.A_tilde, curr_emb)
            embs_list.append(curr_emb)

        stacked_embs = torch.stack(embs_list, dim=1)

        e = torch.mean(stacked_embs, dim=1)
        users_emb, items_emb = torch.split(e, [self.n_users, self.n_items])

        users_idx = batch[:, 0]
        items_idx = batch[:, 1]

        batch_users_emb = users_emb[users_idx]
        batch_items_emb = items_emb[items_idx]

        scores_matrix = batch_users_emb @ batch_items_emb.T

        scores = torch.diagonal(scores_matrix, 0)

        return scores
