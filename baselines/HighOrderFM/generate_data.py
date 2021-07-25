train90_file = "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/train90/CIL_data90.train"
train90aug_file = "/home/ico/PycharmProjects/CIL-2021/baselines/HighOrderFM/CIL_data90aug.train.libfm"
train100_file = "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/train100/CIL_data100.train"
train100aug_file = "/home/ico/PycharmProjects/CIL-2021/baselines/HighOrderFM/CIL_data100aug.train.libfm"
valid_file = "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/valid/CIL_data10.valid"
validaug_file = "/home/ico/PycharmProjects/CIL-2021/baselines/HighOrderFM/CIL_dataaug.valid.libfm"
sub_file = "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/submission/CIL_data.submission"
subaug_file = "/home/ico/PycharmProjects/CIL-2021/baselines/HighOrderFM/CIL_dataaug.submission.libfm"

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
filerows = [1054805, 1176952, 122147, 1176952]
for in_file in [train90_file, train100_file, valid_file, sub_file]:
    print("Processing file", file_idx)

    line_num = 0
    counter = 0
    with open(in_file) as inf:
        #         row = np.ndarray()
        #         col = np.ndarray()
        #         data = np.ndarray()
        #         y = np.ndarray()
        for in_line in inf.readlines():
            split_line = in_line.split("\t")
            user = int(split_line[0]) - 1
            item = int(split_line[1]) - 1
            rating = float(split_line[2][:-1])

            #             y.append(rating)

            #             row.append(line_num)
            #             col.append(user)
            #             data.append(1)

            #             row.append(line_num)
            #             col.append(10000+item)
            #             data.append(1)

            counter += 2

            bagofitems_set = user_bagofitems_dict[str(user + 1)]
            items_rated_inverse = 1 / len(bagofitems_set)
            # for it in bagofitems_set:
                #                 row.append(line_num)
                #                 col.append(11000+int(it)-1)
                #                 data.append(items_rated_inverse)
                # counter += 1
            counter += len(bagofitems_set)

            bagofusers_set = item_bagofusers_dict[str(item + 1)]
            users_rated_inverse = 1 / len(bagofusers_set)
            # for us in bagofusers_set:
                #                 row.append(line_num)
                #                 col.append(12000+int(us)-1)
                #                 data.append(users_rated_inverse)
                # counter += 1
            counter += len(bagofusers_set)

            line_num += 1
            if line_num % 100000 == 0:
                print(line_num)
    print("COUNTER:", counter)

    #     if file_idx == 1:
    #         train90_X = csr_matrix((data, (row, col)), shape=(line_num, 22000))
    #         train90_y = y
    #     elif file_idx == 2:
    #         train100_X = csr_matrix((data, (row, col)), shape=(line_num, 22000))
    #         train100_y = y
    #     elif file_idx == 3:
    #         val10_X = csr_matrix((data, (row, col)), shape=(line_num, 22000))
    #         val10_y = y
    #     elif file_idx == 4:
    #         sub_X = csr_matrix((data, (row, col)), shape=(line_num, 22000))

    file_idx += 1

