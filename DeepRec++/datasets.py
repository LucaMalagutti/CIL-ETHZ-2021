import pickle

import numpy as np
import pandas as pd
import torch


class EmbeddingDataset(torch.utils.data.Dataset):
    """
    Dataset of [user_embedding, item_embedding, user_item_rating]
    """

    def __init__(self, path_to_user_embs, path_to_item_embs, path_to_train_data):
        super().__init__()
        self.delimiter = "\t"

        with open(path_to_user_embs, "rb") as f:
            user_embs = pickle.load(f)
        with open(path_to_item_embs, "rb") as f:
            item_embs = pickle.load(f)

        self.data = []
        self.ratings = []
        with open(path_to_train_data, "r") as src:
            for line in src.readlines():
                parts = line.strip().split(self.delimiter)
                if len(parts) < 3:
                    raise ValueError(
                        "Encountered badly formatted line in {}".format(
                            path_to_train_data
                        )
                    )
                self.data.append(
                    np.array(
                        user_embs[int(parts[0])] + item_embs[int(parts[1])],
                        dtype=np.float32,
                    )
                )
                self.ratings.append(float(parts[2]))
        self.ratings = np.array(self.ratings, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.ratings[idx]


class RatingDataset(torch.utils.data.Dataset):
    """
    Dataset of [user_matrix_vector, item_matrix_vector, user_item_rating]
    """

    def __init__(self, path_to_train_data):
        super().__init__()
        self.path_to_train_data = path_to_train_data
        self.delimiter = "\t"
        self.train_matrix = self._create_train_data_matrix()

        self.user_vectors = []
        self.item_vectors = []
        self.ratings = []
        with open(self.path_to_train_data, "r") as src:
            for line in src.readlines():
                parts = line.strip().split(self.delimiter)
                if len(parts) < 3:
                    raise ValueError(
                        "Encountered badly formatted line in {}".format(
                            path_to_train_data
                        )
                    )
                self.user_vectors.append(self.train_matrix[int(parts[0]) - 1])
                self.item_vectors.append(self.train_matrix[:, int(parts[1]) - 1])
                self.ratings.append(int(float(parts[2])))

    def _create_train_data_matrix(self):
        train_df = pd.read_csv(self.path_to_train_data, delimiter="\t", header=None)
        train_df[2] = train_df[2].astype("int32")

        NUM_ITEMS = 1000
        NUM_USERS = 10000

        train_matrix = np.zeros((NUM_USERS, NUM_ITEMS))

        for user_id, item_id, rating in train_df.to_numpy(dtype=np.int32):
            train_matrix[user_id - 1][item_id - 1] = int(float(rating))

        return train_matrix

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_vectors[idx], self.item_vectors[idx], self.ratings[idx]
