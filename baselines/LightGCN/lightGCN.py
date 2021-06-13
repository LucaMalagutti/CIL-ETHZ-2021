import os

import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class LightGCN(nn.Module):
    def __init__(self, args):
        super(LightGCN, self).__init__()
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=10000, embedding_dim=args.emb_size
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=1000, embedding_dim=args.emb_size
        )

        if args.restore_ckpt is None:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)

        # A_tilde = build_A_tilde()

    def build_A_tilde(self):
        pass

    def forward(self):
        pass
