import json

train90_file = "/baselines/DeepRec/data/train90/CIL_data90.train"
train90aug_file = "data_features/deepf1stb_bags/CIL_data90aug.train.libfm"
train100_file = "/baselines/DeepRec/data/train100/CIL_data100.train"
train100aug_file = "data_features/deepf1stb_bags/CIL_data100aug.train.libfm"
valid_file = "/baselines/DeepRec/data/valid/CIL_data10.valid"
validaug_file = "data_features/deepf1stb_bags/CIL_dataaug.valid.libfm"
sub_file = "/baselines/DeepRec/data/submission/CIL_data.submission"
subaug_file = "data_features/deepf1stb_bags/CIL_dataaug.submission.libfm"
deepf_json =        "/home/ico/PycharmProjects/CIL-2021/baselines/BayesianSVD/data_features/deepfeatures/deepfeatures100_1stBottleneck.json"
deepf90_json =      "/home/ico/PycharmProjects/CIL-2021/baselines/BayesianSVD/data_features/deepfeatures/deepfeatures90_1stBottleneck.json"

user_bagofitems_dict = dict()
item_bagofusers_dict = dict()



# fill bags dictionaries
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

# file generation loop
file_idx = 1
for in_file, out_file in zip([train90_file, train100_file, valid_file, sub_file], [train90aug_file, train100aug_file, validaug_file, subaug_file]):
    if file_idx <= 2:
        ext = ".train"
    else:
        ext = ".test"

    if file_idx == 1 or file_idx == 3:
        curr_deepf_json = deepf90_json
    else:
        curr_deepf_json = deepf_json

    with open(curr_deepf_json) as deepf_file:
        deepf_dict = json.load(deepf_file)

        print("Processing file", file_idx)
        file_idx += 1

        # rel_user.libfm file
        curr_out_file = out_file + "rel_user.libfm"
        with open(curr_out_file, "w") as outf:
            line_num = 0
            for user in range(10000):
                # generate bag features string (indexes from 10000 to 10999)
                bagofitems_set = user_bagofitems_dict[str(user + 1)]
                items_rated_inverse = str(1 / len(bagofitems_set))
                bagofitems_str = ""
                for item in bagofitems_set:
                    bagofitems_str += (" " + str(int(item)+9999) + ":" + items_rated_inverse)

                # generate deepf string (indexes from 11000 to 11031
                deepf_vector = deepf_dict[str(user + 1)]
                deepf_str = ""
                i = 11000
                for deepf in deepf_vector:
                    deepf_str += (" " + str(i) + ":" + deepf)
                    i += 1

                user = str(user)
                outf.write("0 " + user + ":1" + bagofitems_str + deepf_str + "\n")

                line_num += 1
                if line_num % 100000 == 0:
                    print(line_num)

        # rel_item.libfm file
        curr_out_file = out_file + "rel_item.libfm"
        with open(curr_out_file, "w") as outf:
            line_num = 0
            for item in range(1000):
                # generate bag features string (indexes from 1000 to 10999)
                bagofusers_set = item_bagofusers_dict[str(item + 1)]
                users_rated_inverse = str(1 / len(bagofusers_set))
                bagofusers_str = ""
                for user in bagofusers_set:
                    bagofusers_str += (" " + str(int(user) + 999) + ":" + users_rated_inverse)

                # generate deepf of items str

                item = str(item)
                outf.write("0 " + item + ":1" + bagofusers_str + "\n")

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

            # deep features = group 2
            for i in range(32):
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

            # item deepf = group 2

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
