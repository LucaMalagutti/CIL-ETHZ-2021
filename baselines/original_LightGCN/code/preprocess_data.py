from numpy.core.fromnumeric import mean
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import world
import scipy.sparse as sp

number_of_users, number_of_movies = (10000, 1000)
data_pd = pd.read_csv('../../../data/data_train.csv')

# Split the dataset into train and test
train_size = 0.9
train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=world.seed)

mean_train = np.mean(train_pd.Prediction.values)
mean_test = np.mean(test_pd.Prediction.values)


def extract_users_items_predictions(data_pd):
    users, movies = [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    ratings = data_pd.Prediction.values
    return users, movies, ratings


train_users, train_movies, train_ratings = extract_users_items_predictions(train_pd)
test_users, test_movies, test_ratings = extract_users_items_predictions(test_pd)

# define and save train dataframe
column_names = ['user', 'movie', 'rating']
train_dataset = np.column_stack((train_users, train_movies, train_ratings))
train_df = pd.DataFrame(data=train_dataset)

train_df.columns = column_names
train_df.to_csv("../data/cil/train.csv", index=False)

# define and save test dataframe
test_dataset = np.column_stack((test_users, test_movies, test_ratings))
test_df = pd.DataFrame(data=test_dataset)
test_df.columns = column_names
test_df.to_csv("../data/cil/test.csv", index=False)

# create and save full training matrix of observed ratings
filled_training_matrix = np.full((number_of_users, number_of_movies), mean_train)
training_mask = np.full((number_of_users, number_of_movies), 0)
for user, movie, rating in zip(train_users, train_movies, train_ratings):
    filled_training_matrix[user][movie] = rating
    training_mask[user][movie] = 1

sp.save_npz("../data/cil/R_train", sp.csr_matrix(filled_training_matrix))
sp.save_npz("../data/cil/R_mask_train", sp.csr_matrix(filled_training_matrix))

# create and save full testing matrix of observed ratings
filled_testing_matrix = np.full((number_of_users, number_of_movies), mean_train*train_size+mean_test*(1-train_size))
test_mask = np.full((number_of_users, number_of_movies), 0)
for user, movie, rating in zip(test_users, test_movies, test_ratings):
    filled_testing_matrix[user][movie] = rating
    test_mask[user][movie] = 1

sp.save_npz("../data/cil/R_test", sp.csr_matrix(filled_testing_matrix))
sp.save_npz("../data/cil/R_mask_test", sp.csr_matrix(test_mask))



