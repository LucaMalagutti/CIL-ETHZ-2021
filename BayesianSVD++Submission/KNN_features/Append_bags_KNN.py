"""
Generates and appends KNN features to LibFM BayesianSVD++ train, valid and submission files
"""

train90_file = "/baselines/DeepRec/data/train90/CIL_data90.train"
train90aug_file = "data_features/KNN_features/CIL_data90aug.train.libfm"
train100_file = "/baselines/DeepRec/data/train100/CIL_data100.train"
train100aug_file = "data_features/KNN_features/CIL_data100aug.train.libfm"
valid_file = "/baselines/DeepRec/data/valid/CIL_data10.valid"
validaug_file = "data_features/KNN_features/CIL_dataaug.valid.libfm"
sub_file = "/baselines/DeepRec/data/submission/CIL_data.submission"
subaug_file = "data_features/KNN_features/CIL_dataaug.submission.libfm"

user_bagofitems_dict = dict()
item_bagofusers_dict = dict()

for in_file in [train100_file, sub_file]:
    with open(in_file) as inf:
        for in_line in inf.readlines():
            split_line = in_line.split("\t")
            user = split_line[0]
            item = split_line[1]
            if user not in user_bagofitems_dict:
                # new user: create new set of items rated by the user
                user_bagofitems_dict[user] = set()
            user_bagofitems_dict[user].add(item)
            if item not in item_bagofusers_dict:
                # new item: create new set of users that rated the item
                item_bagofusers_dict[item] = set()
            item_bagofusers_dict[item].add(user)

# KNN features

user_allratings100_dict = dict()
item_allratings100_dict = dict()

with open(train100_file) as inf:
    for in_line in inf.readlines():
        split_line = in_line.split("\t")
        user = split_line[0]
        item = split_line[1]
        rating = split_line[2][:-1]
        if user not in user_allratings100_dict:
            # new user: create new set of (item, rating) rated by the user
            user_allratings100_dict[user] = set()
        user_allratings100_dict[user].add((item, rating))
        if item not in item_allratings100_dict:
            # new item: create new set of (user, rating) that rated the item
            item_allratings100_dict[item] = set()
        item_allratings100_dict[item].add((user, rating))

user_allratings90_dict = dict()
item_allratings90_dict = dict()

with open(train90_file) as inf:
    for in_line in inf.readlines():
        split_line = in_line.split("\t")
        user = split_line[0]
        item = split_line[1]
        rating = split_line[2][:-1]
        if user not in user_allratings90_dict:
            # new user: create new set of (item, rating) rated by the user
            user_allratings90_dict[user] = set()
        user_allratings90_dict[user].add((item, rating))
        if item not in item_allratings90_dict:
            # new item: create new set of (user, rating) that rated the item
            item_allratings90_dict[item] = set()
        item_allratings90_dict[item].add((user, rating))

file_idx = 1
for in_file, out_file in zip([train90_file, train100_file, valid_file, sub_file], [train90aug_file, train100aug_file, validaug_file, subaug_file]):
    if file_idx <= 2:
        ext = ".train"
    else:
        ext = ".test"
    print("Processing file", file_idx)
    file_idx += 1

    # rel_user.libfm file
    curr_out_file = out_file + "rel_user.libfm"
    with open(curr_out_file, "w") as outf:
        line_num = 0
        for user in range(10000):
            bagofitems_set = user_bagofitems_dict[str(user + 1)]
            items_rated_inverse = str(1 / len(bagofitems_set))
            bagofitems_str = ""
            for item in bagofitems_set:
                bagofitems_str += (" " + str(int(item)+9999) + ":" + items_rated_inverse)

            # KNN features
            if in_file == train90_file or in_file == valid_file:
                allratings_set = user_allratings90_dict[str(user + 1)]
            else:
                allratings_set = user_allratings100_dict[str(user + 1)]
            allratings_str = ""
            for item, rating in allratings_set:
                allratings_str += (" " + str(int(item)+10999) + ":" + str(float(rating)/1000))

            user = str(user)
            outf.write("0 " + user + ":1" + bagofitems_str + allratings_str + "\n")

            line_num += 1
            if line_num % 100000 == 0:
                print(line_num)

    # rel_item.libfm file
    curr_out_file = out_file + "rel_item.libfm"
    with open(curr_out_file, "w") as outf:
        line_num = 0
        for item in range(1000):
            bagofusers_set = item_bagofusers_dict[str(item + 1)]
            users_rated_inverse = str(1 / len(bagofusers_set))
            bagofusers_str = ""
            for user in bagofusers_set:
                bagofusers_str += (" " + str(int(user) + 999) + ":" + users_rated_inverse)

            # KNN features
            if in_file == train90_file or in_file == valid_file:
                allratings_set = item_allratings90_dict[str(item + 1)]
            else:
                allratings_set = item_allratings100_dict[str(item + 1)]
            allratings_str = ""
            for user, rating in allratings_set:
                allratings_str += (" " + str(int(user) + 10999) + ":" + str(float(rating) / 10000))

            item = str(item)
            outf.write("0 " + item + ":1" + bagofusers_str + allratings_str + "\n")

            line_num += 1
            if line_num % 100000 == 0:
                print(line_num)

    # rel_user.train / .test file
    curr_out_file = out_file + "rel_user" + ext
    with open(curr_out_file, "w") as outf:
        line_num = 0
        with open(in_file) as inf:
            for in_line in inf.readlines():
                split_line = in_line.split("\t")
                user = split_line[0]
                user = str(int(user) - 1)               # users have ids between 0 and 9999

                outf.write(user + "\n")

                line_num += 1
                if line_num % 100000 == 0:
                    print(line_num)

    # rel_item.train / .test file
    curr_out_file = out_file + "rel_item" + ext
    with open(curr_out_file, "w") as outf:
        line_num = 0
        with open(in_file) as inf:
            for in_line in inf.readlines():
                split_line = in_line.split("\t")
                item = split_line[1]
                item = str(int(item) - 1)  # items have ids between 0 and 999

                outf.write(item + "\n")

                line_num += 1
                if line_num % 100000 == 0:
                    print(line_num)

    # rel_user.groups
    curr_out_file = out_file + "rel_user.groups"
    with open(curr_out_file, "w") as outf:
        line_num = 0

        # users = group 0
        for i in range(10000):
            outf.write("0\n")

            line_num += 1
            if line_num % 100000 == 0:
                print(line_num)

        # bag of items = group 1
        for i in range(1000):
            outf.write("1\n")

            line_num += 1
            if line_num % 100000 == 0:
                print(line_num)

        # KNN features (ratings given to other items) = group 2
        for i in range(1000):
            outf.write("2\n")

            line_num += 1
            if line_num % 100000 == 0:
                print(line_num)

    # rel_item.groups (if bag of users is added)
    curr_out_file = out_file + "rel_item.groups"
    with open(curr_out_file, "w") as outf:
        line_num = 0

        # items = group 0
        for i in range(1000):
            outf.write("0\n")

            line_num += 1
            if line_num % 100000 == 0:
                print(line_num)

        # bag of users = group 1
        for i in range(10000):
            outf.write("1\n")

            line_num += 1
            if line_num % 100000 == 0:
                print(line_num)

        # KNN features (ratings given by other users) = group 2
        for i in range(10000):
            outf.write("2\n")

            line_num += 1
            if line_num % 100000 == 0:
                print(line_num)

    # y.train / .test file
    curr_out_file = out_file + "y" + ext
    with open(curr_out_file, "w") as outf:
        line_num = 0
        with open(in_file) as inf:
            for in_line in inf.readlines():
                split_line = in_line.split("\t")
                rating = split_line[2][:-1]

                outf.write(rating + "\n")

                line_num += 1
                if line_num % 100000 == 0:
                    print(line_num)
