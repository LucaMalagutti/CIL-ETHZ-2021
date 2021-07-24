train90_file = "/baselines/DeepRec/data/train90/CIL_data90.train"
train90hofm_file = "/baselines/BayesianSVD/data_features/bags_HOFM/HOFMtrain90.csv"
train90aug_file = "data_features/bags_HOFM/CIL_data90aug.train.libfm"
train100_file = "/baselines/DeepRec/data/train100/CIL_data100.train"
train100hofm_file = "/baselines/BayesianSVD/data_features/bags_HOFM/HOFMtrain100.csv"
train100aug_file = "data_features/bags_HOFM/CIL_data100aug.train.libfm"
valid_file = "/baselines/DeepRec/data/valid/CIL_data10.valid"
validhofm_file = "/baselines/BayesianSVD/data_features/bags_HOFM/HOFMval10.csv"
validaug_file = "data_features/bags_HOFM/CIL_dataaug.valid.libfm"
sub_file = "/baselines/DeepRec/data/submission/CIL_data.submission"
subhofm_file = "/baselines/BayesianSVD/data_features/bags_HOFM/HOFMsub.csv"
subaug_file = "data_features/bags_HOFM/CIL_dataaug.submission.libfm"

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

file_idx = 1
for in_file, out_file, hofm_file in zip([train90_file, train100_file, valid_file, sub_file],
                                        [train90aug_file, train100aug_file, validaug_file, subaug_file],
                                        [train90hofm_file, train100hofm_file, validhofm_file, subhofm_file]):
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

            user = str(user)
            outf.write("0 " + user + ":1" + bagofitems_str + "\n")

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

            item = str(item)
            outf.write("0 " + item + ":1" + bagofusers_str + "\n")

            line_num += 1
            if line_num % 100000 == 0:
                print(line_num)

    # rel_hofm.libfm file
    curr_out_file = out_file + "rel_hofm.libfm"
    with open(curr_out_file, "w") as outf:
        line_num = 0
        with open(hofm_file) as hf:
            for hf_line in hf.readlines():
                hf_rating = hf_line[:-1]
                outf.write("0 " + "0:" + hf_rating + " 1:0" + "\n")

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

    # rel_hofm.train / .test file
    curr_out_file = out_file + "rel_hofm" + ext
    with open(curr_out_file, "w") as outf:
        line_num = 0
        with open(in_file) as inf:
            for in_line in inf.readlines():

                outf.write(str(line_num) + "\n")

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
