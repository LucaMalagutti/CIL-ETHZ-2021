import json

train90_file = "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/train90/CIL_data90.train"
train90aug_file = "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/deepfeatures/CIL_data90aug.train"
train100_file = "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/train100/CIL_data100.train"
train100aug_file = "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/deepfeatures/CIL_data100aug.train"
valid_file = "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/valid/CIL_data.valid"
validaug_file = "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/deepfeatures/CIL_dataaug.valid"
sub_file = "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/submission/CIL_data.submission"
subaug_file = "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/deepfeatures/CIL_dataaug.submission"
deepf_json = "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/deepfeatures/deepfeatures.json"

with open(deepf_json) as deepf_file:
    deepf_dict = json.load(deepf_file)

    for in_file, out_file in zip([train90_file, train100_file, valid_file, sub_file], [train90aug_file, train100aug_file, validaug_file, subaug_file]):
        with open(in_file) as inf:
            with open(out_file, "w") as outf:
                line_num = 0
                for in_line in inf.readlines():
                    user = in_line.split("\t")[0]
                    deepf_vector = deepf_dict[user]
                    deepf_str = "\t".join(deepf_vector)

                    # print("writing:",in_line[:-1]+ "\t" + deepf_str)
                    outf.write(in_line[:-1] + "\t" + deepf_str)

                    line_num += 1
                    if line_num % 100000 == 0:
                        print(line_num)
