"""Transforms the CIL competition dataset into a form usable by the autoencoder model"""

import random
import sys
from math import floor
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)


def print_stats(data):
    total_ratings = 0
    print("STATS")
    for user in data:
        total_ratings += len(data[user])
    items_set = []
    for it_ratings_list in data.values():
        items_list = [entry[0] for entry in it_ratings_list]
        items_set.extend(items_list)
    items_set = set(items_set)
    print("Total Ratings: {}".format(total_ratings))
    print("Total User count: {}".format(len(data.keys())))
    print("Total items count: {}".format(len(items_set)))


def save_data_to_file(data, filename):
    with open(filename, "w") as out:
        for item in data:
            for user, rating in data[item]:
                out.write("{}\t{}\t{}\n".format(user, item, rating))


def convert2CILdictionary(dictionary):
    """
    Converts dictionary to newdictionary with items as keys, (user, rating) tuples as values.
    Sorts the items and for each item sorts the (user, rating) tuples by user.
    @param dictionary: dictionary with users as keys, (item, rating) tuples as values.
    """
    newdictionary = dict()
    for user in dictionary:
        for item, rating in dictionary[user]:
            if item not in newdictionary:
                newdictionary[item] = []
            newdictionary[item].append((user, rating))

    # sort by item, then by user, as in the original csv
    for item in newdictionary:
        newdictionary[item] = sorted(newdictionary[item])
    return dict(sorted(newdictionary.items()))


def main(args):
    inpt = args[1]
    out_prefix_train = "data/train90/CIL_data90"
    out_prefix_valid = "data/valid/CIL_data10"
    out_prefix_submission = "data/submission/CIL_data"

    Path(out_prefix_train).mkdir(parents=True, exist_ok=True)
    Path(out_prefix_valid).mkdir(parents=True, exist_ok=True)
    Path(out_prefix_submission).mkdir(parents=True, exist_ok=True)

    # 0.9 for 90%, 1.0 for 100% train and no validation
    percent = 0.9
    if len(args) > 2:
        if args[2] == "submission":
            # take all the ratings to generate the submission file
            percent = 1

    data = dict()

    total_rating_count = 0
    with open(inpt, "r") as inpt_f:  # ratings.csv headers: userId,movieId,rating
        for line in inpt_f:
            if "Id" in line:
                continue
            parts = line.split(",")
            useritem = parts[0].split("_")
            user = int(useritem[0][1:])
            item = int(useritem[1][1:])
            rating = float(parts[1])

            total_rating_count += 1
            if user not in data:
                data[user] = []
            data[user].append((item, rating))

    print("STATS")
    print("Total Ratings: {}".format(total_rating_count))

    training_data = dict()
    validation_data = dict()
    train_set_items = set()

    random.seed(1235)

    for user in data.keys():
        if len(data[user]) < 2:
            print(
                "WARNING, userId {} has less than 2 ratings, skipping user...".format(
                    user
                )
            )
            continue
        ratings = data[user]
        if len(args) <= 2:
            random.shuffle(ratings)
        last_train_ind = floor(percent * len(ratings))
        training_data[user] = ratings[:last_train_ind]
        for rating_item in ratings[:last_train_ind]:
            train_set_items.add(rating_item[0])  # keep track of items from training set

        validation_data[user] = ratings[last_train_ind:]

    # remove items not not seen in training set
    for user, userRatings in validation_data.items():
        validation_data[user] = [
            rating for rating in userRatings if rating[0] in train_set_items
        ]

    # Saves train and evaluation data files
    if len(args) <= 2:
        print("Training Data")
        print_stats(training_data)
        save_data_to_file(
            convert2CILdictionary(training_data), out_prefix_train + ".train"
        )
        print("Validation Data")
        print_stats(validation_data)
        save_data_to_file(
            convert2CILdictionary(validation_data), out_prefix_valid + ".valid"
        )
    # Saves submission data file
    elif args[2] == "submission":
        print("Submission Data")
        print_stats(training_data)
        save_data_to_file(
            convert2CILdictionary(training_data), out_prefix_submission + ".submission"
        )
    else:
        print("Invalid arguments:", args[2])


if __name__ == "__main__":
    main(sys.argv)
