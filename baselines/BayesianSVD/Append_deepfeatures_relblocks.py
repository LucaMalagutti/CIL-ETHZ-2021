import json

train90_file =      "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/train90/CIL_data90.train"
train90aug_file =   "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/deepfeatures/CIL_data90aug.train.libfm"
train100_file =     "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/train100/CIL_data100.train"
train100aug_file =  "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/deepfeatures/CIL_data100aug.train.libfm"
valid_file =        "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/valid/CIL_data.valid"
validaug_file =     "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/deepfeatures/CIL_dataaug.valid.libfm"
sub_file =          "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/submission/CIL_data.submission"
subaug_file =       "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/deepfeatures/CIL_dataaug.submission.libfm"
deepf_json =        "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/deepfeatures/deepfeatures.json"

with open(deepf_json) as deepf_file:
    deepf_dict = json.load(deepf_file)

    file_idx = 1
    for in_file, out_file in zip([train90_file, train100_file, valid_file, sub_file], [train90aug_file, train100aug_file, validaug_file, subaug_file]):
        print("Processing file", file_idx)
        file_idx += 1

        # .x file
        out_file += ".x"
        with open(out_file, "w") as outf:
            line_num = 0
            with open(in_file) as inf:
                for in_line in inf.readlines():
                    split_line = in_line.split("\t")
                    user = split_line[0]
                    # item = str(int(split_line[1]) + 9999)  # items have ids between 10000 and 10999
                    # rating = split_line[2][:-1]             # skip "\n"
                    deepf_vector = deepf_dict[user]
                    deepf_str = ""
                    i = 10000
                    for deepf in deepf_vector:
                        deepf_str += (" " + str(i) + ":" + deepf)
                        i += 1

                    user = str(int(user) - 1)               # users have ids between 0 and 9999
                    outf.write(rating + " " + user + ":1 " + item + ":1" + deepf_str + "\n")

                    line_num += 1
                    if line_num % 100000 == 0:
                        print(line_num)
