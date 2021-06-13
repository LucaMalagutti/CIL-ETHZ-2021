import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.core.fromnumeric import mean
from sklearn.model_selection import train_test_split

number_of_users, number_of_movies = (10000, 1000)
data_pd = pd.read_csv("../../data/data_train.csv")

# Split the dataset into train and val
train_size = 0.9
train_pd, val_pd = train_test_split(data_pd, train_size=train_size, random_state=2021)

mean_train = np.mean(train_pd.Prediction.values)
mean_val = np.mean(val_pd.Prediction.values)


def extract_users_items_predictions(data_pd):
    reg = r"r(\d+)_c(\d+)"

    users, movies = [
        np.squeeze(arr)
        for arr in np.split(
            data_pd.Id.str.extract(reg).values.astype(int) - 1, 2, axis=-1
        )
    ]
    ratings = data_pd.Prediction.values
    return users, movies, ratings


train_users, train_movies, train_ratings = extract_users_items_predictions(train_pd)
val_users, val_movies, val_ratings = extract_users_items_predictions(val_pd)

# define and save train dataframe
column_names = ["user", "movie", "rating"]
train_dataset = np.column_stack((train_users, train_movies, train_ratings))
train_df = pd.DataFrame(data=train_dataset)

train_df.columns = column_names
train_df.to_csv("data/train.csv", index=False)

# define and save val dataframe
val_dataset = np.column_stack((val_users, val_movies, val_ratings))
val_df = pd.DataFrame(data=val_dataset)
val_df.columns = column_names
val_df.to_csv("data/val.csv", index=False)

# create and save full training matrix of observed ratings
filled_training_matrix = np.full((number_of_users, number_of_movies), 0)
training_mask = np.full((number_of_users, number_of_movies), 0)
for user, movie, rating in zip(train_users, train_movies, train_ratings):
    filled_training_matrix[user][movie] = rating
    training_mask[user][movie] = 1

sp.save_npz("data/R_train", sp.csr_matrix(filled_training_matrix))
sp.save_npz("data/R_mask_train", sp.csr_matrix(filled_training_matrix))

# create and save full validation matrix of observed ratings
filled_validation_matrix = np.full((number_of_users, number_of_movies), 0)
val_mask = np.full((number_of_users, number_of_movies), 0)
for user, movie, rating in zip(val_users, val_movies, val_ratings):
    filled_validation_matrix[user][movie] = rating
    val_mask[user][movie] = 1

sp.save_npz("data/R_val", sp.csr_matrix(filled_validation_matrix))
sp.save_npz("data/R_mask_val", sp.csr_matrix(val_mask))
