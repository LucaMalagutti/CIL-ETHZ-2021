"""Contains the method that generates the submission .csv file in the 'submissions' folder"""


import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from preprocessing import *
from models import *
from util_functions import *
from torch_geometric.data import DataLoader


def generate_submission(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_state_dict = torch.load(args.restore_ckpt)

    # prepare submission dataset: delete folder data/CIL/submission before running this script
    sub_pd = pd.read_csv("../../data/sample_submission.csv")
    sub_users, sub_movies, sub_ratings = extract_users_items_predictions(sub_pd)
    sub_dataset = np.column_stack((sub_users, sub_movies, sub_ratings))
    sub_data = pd.DataFrame(data=sub_dataset)
    sub_data.columns = ["user", "movie", "rating"]
    sub_indices = (sub_users, sub_movies)
    sub_labels = sub_ratings
    u_features = None
    v_features = None
    class_values = np.array([1, 2, 3, 4, 5])

    # compute adj_train
    rating_map = {x: int(x) for x in np.arange(1, 5.01, 1).tolist()}
    post_rating_map = {
        x: int(i+1 // (5 / args.num_relations))
        for i, x in enumerate(np.arange(1, 6).tolist())
    }
    adj_train, train_labels, train_u_indices, train_v_indices = create_CIL_trainvaltest_split(False, 1234, True, rating_map, post_rating_map, args.ratio, 0)[2:6]
    train_indices = (train_u_indices, train_v_indices)
    val_test_appendix = 'submission'
    data_combo = (args.data_name, args.data_appendix, val_test_appendix)
    data_combo_train = (args.data_name, args.data_appendix, 'valmode')

    # create train graphs
    train_graphs = MyDataset(
        'data/{}{}/{}/train'.format(*data_combo_train),
        adj_train,
        train_indices,
        train_labels,
        args.hop,
        args.sample_ratio,
        args.max_nodes_per_hop,
        u_features,
        v_features,
        class_values,
        max_num=args.max_train_num
    )

    # create submission graphs
    sub_graphs = MyDataset(
        'data/{}{}/{}/test'.format(*data_combo),
        adj_train,
        sub_indices,
        sub_labels,
        args.hop,
        args.sample_ratio,
        args.max_nodes_per_hop,
        u_features,
        v_features,
        class_values,
        max_num=args.max_test_num
    )

    model = IGMC(
        train_graphs,
        latent_dim=[32, 32, 32, 32],
        num_relations=args.num_relations,
        num_bases=4,
        regression=True,
        adj_dropout=args.adj_dropout,
        force_undirected=args.force_undirected,
        side_features=args.use_features,
        n_side_features=0,
        multiply_by=args.multiply_by
    )
    model.load_state_dict(saved_state_dict)
    sub_loader = DataLoader(sub_graphs, 1, shuffle=False)
    model.to(device)
    model.eval()
    scores_list = []

    for data in tqdm(sub_loader):
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            scores_list.extend(out.tolist())
        torch.cuda.empty_cache()

    assert len(scores_list) == len(sub_data["rating"])

    sub_data["Prediction"] = scores_list
    sub_data["Id"] = (
        "r"
        + sub_data["user"].apply(lambda x: str(x + 1))
        + "_"
        + "c"
        + sub_data["movie"].apply(lambda x: str(x + 1))
    )
    sub_data = sub_data[["Id", "Prediction"]]

    sub_name = 'IGMC_sub.csv'
    print('sub_name:', sub_name)
    sub_data.to_csv(os.path.join("submissions", sub_name), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--restore_ckpt",
        type=str,
        default=None,
        help="restore model weights from checkpoint",
        required=True,
    )
    parser.add_argument('--hop', default=1, metavar='S',
                        help='enclosing subgraph hop number')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                        help='if < 1, subsample nodes per hop according to the ratio')
    parser.add_argument('--max-nodes-per-hop', default=100,
                        help='if > 0, upper bound the # nodes per hop by another subsampling')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help="For ml datasets, if ratio < 1, downsample training data to the\
                        target ratio")
    parser.add_argument('--testing', action='store_true', default=False,
                        help='if set, use testing mode which splits all ratings into train/test;\
                        otherwise, use validation model which splits all ratings into \
                        train/val/test and evaluate on val only')
    parser.add_argument('--num-relations', type=int, default=5,
                        help='if transfer, specify num_relations in the transferred model')
    parser.add_argument('--multiply-by', type=int, default=1,
                        help='if transfer, specify how many times to multiply the predictions by')
    parser.add_argument('--adj-dropout', type=float, default=0.2,
                        help='if not 0, random drops edges from adjacency matrix with this prob')
    parser.add_argument('--force-undirected', action='store_true', default=False,
                        help='in edge dropout, force (x, y) and (y, x) to be dropped together')
    parser.add_argument('--use-features', action='store_true', default=False,
                        help='whether to use node features (side information)')
    parser.add_argument('--data-name', default='CIL', help='dataset name')
    parser.add_argument('--data-appendix', default='',
                        help='what to append to save-names when saving datasets')
    parser.add_argument('--max-test-num', type=int, default=None,
                        help='set maximum number of test data to use')
    parser.add_argument('--max-train-num', type=int, default=None,
                        help='set maximum number of train data to use')

    args = parser.parse_args()

    torch.manual_seed(2021)
    np.random.seed(2021)

    if not os.path.isdir("submissions"):
        os.mkdir("submissions")

    generate_submission(args)
