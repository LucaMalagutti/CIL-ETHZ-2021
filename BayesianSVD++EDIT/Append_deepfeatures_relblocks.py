import json

train90_file = "/baselines/DeepRec/data/train90/CIL_data90.train"
train90aug_file = "/baselines/BayesianSVD/data_features/deepfeatures/CIL_data90aug.train.libfm"
train100_file = "/baselines/DeepRec/data/train100/CIL_data100.train"
train100aug_file = "/baselines/BayesianSVD/data_features/deepfeatures/CIL_data100aug.train.libfm"
valid_file = "/baselines/DeepRec/data/valid/CIL_data10.valid"
validaug_file = "/baselines/BayesianSVD/data_features/deepfeatures/CIL_dataaug.valid.libfm"
sub_file = "/baselines/DeepRec/data/submission/CIL_data.submission"
subaug_file = "/baselines/BayesianSVD/data_features/deepfeatures/CIL_dataaug.submission.libfm"
deepf_json =        "/home/ico/PycharmProjects/CIL-2021/baselines/BayesianSVD/data_features/deepfeatures/deepfeatures.json"

with open(deepf_json) as deepf_file:
    deepf_dict = json.load(deepf_file)

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
            # with open(in_file) as inf:
            # for in_line in inf.readlines():
            for user in range(10000):
                # split_line = in_line.split("\t")
                # user = split_line[0]
                # item = str(int(split_line[1]) + 9999)  # items have ids between 10000 and 10999
                # rating = split_line[2][:-1]             # skip "\n"
                deepf_vector = deepf_dict[str(user+1)]
                deepf_str = ""
                i = 10000
                for deepf in deepf_vector:
                    deepf_str += (" " + str(i) + ":" + deepf)
                    i += 1

                # user = str(int(user) - 1)               # users have ids between 0 and 9999
                user = str(user)
                # outf.write(rating + " " + user + ":1 " + item + ":1" + deepf_str + "\n")
                outf.write("0 " + user + ":1" + deepf_str + "\n")

                line_num += 1
                if line_num % 100000 == 0:
                    print(line_num)

        # rel_item.libfm file
        curr_out_file = out_file + "rel_item.libfm"
        with open(curr_out_file, "w") as outf:
            line_num = 0
            for item in range(1000):
                item = str(item)
                outf.write("0 " + item + ":1\n")

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

            # deep features = group 1
            for i in range(32):
                outf.write("1\n")

                line_num += 1
                if line_num % 100000 == 0:
                    print(line_num)

        # rel_item.groups (if bag of users is added)

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
