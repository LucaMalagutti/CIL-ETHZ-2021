"""
Generates ensemble predictions on the submission set,
given level 1 model predictions on the validation and submission set
"""

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge

TARGET = 'y'

train_features = None
valid_features = None
y_train = None

train90_file =      "../../baselines/BayesianSVD/data_split/train90/CIL_data90.train"
train100_file =     "../../baselines/BayesianSVD/data_split/train100/CIL_data100.train"
valid_file =        "../../baselines/BayesianSVD/data_split/valid10/CIL_data10.valid"
sub_file =          "../../baselines/BayesianSVD/data_split/submission/CIL_data.submission"

train_out_file = "train_ensemble.csv"
sub_out_file = "sub_ensemble.csv"

def main(args):
    models_preds = args[1:7]
    models_sub_preds = args[7:13]

    user_nratings_dict = dict()
    item_nratings_dict = dict()

    for in_file in [train100_file, sub_file]:
        with open(in_file) as inf:
            for in_line in inf.readlines():
                split_line = in_line.split("\t")
                user = int(split_line[0])
                item = int(split_line[1])
                if user not in user_nratings_dict:
                    # new user: create new set of items rated by the user
                    user_nratings_dict[user] = 0
                user_nratings_dict[user] += 1
                if item not in item_nratings_dict:
                    # new item: create new set of users that rated the item
                    item_nratings_dict[item] = 0
                item_nratings_dict[item] += 1

    train_Xy = np.ndarray((122147, 5+len(models_preds)))
    with open(valid_file, "r") as inf:
        line_num = 0
        for in_line in inf.readlines():
            split_line = in_line.split("\t")
            user = int(split_line[0])
            item = int(split_line[1])
            rating = float(split_line[2][:-1])

            train_Xy[line_num,0] = rating
            train_Xy[line_num,1] = user
            train_Xy[line_num,2] = item
            train_Xy[line_num,3] = user_nratings_dict[user]
            train_Xy[line_num,4] = item_nratings_dict[item]

            line_num += 1

    model_num = 0
    for model_preds in models_preds:
        print("model:",model_num," is:",model_preds)
        with open(model_preds, "r") as inf:
            line_num = 0
            for in_line in inf.readlines():
                pred = float(in_line[:-1])
                train_Xy[line_num, 5 + model_num] = pred
                line_num += 1
        model_num += 1

    test_X = np.ndarray((1176952, 5 + len(models_preds)))
    with open(sub_file, "r") as inf:
        line_num = 0
        for in_line in inf.readlines():
            split_line = in_line.split("\t")
            user = int(split_line[0])
            item = int(split_line[1])
            rating = float(split_line[2][:-1])

            test_X[line_num, 0] = rating
            test_X[line_num, 1] = user
            test_X[line_num, 2] = item
            test_X[line_num, 3] = user_nratings_dict[user]
            test_X[line_num, 4] = item_nratings_dict[item]

            line_num += 1

    model_num = 0
    for model_preds in models_sub_preds:
        print("model:", model_num, " is:", model_preds)
        with open(model_preds, "r") as inf:
            line_num = 0
            for in_line in inf.readlines():
                pred = float(in_line[:-1])
                test_X[line_num, 5 + model_num] = pred
                line_num += 1
        model_num += 1


    columns_dict = {'y': train_Xy[:, 0],
                    'user': train_Xy[:, 1],
                    'item': train_Xy[:, 2],
                    'user_nratings': train_Xy[:, 3],
                    'item_nratings': train_Xy[:, 4]}

    model_num = 1
    print("models_preds:", models_preds)
    for model in models_preds:
        print("adding model:", model)
        columns_dict['model'+str(model_num)] = train_Xy[:, 4 + model_num]
        model_num += 1

    train_df = pd.DataFrame(columns_dict)

    print(train_df)

    columns_dict_test = {'y': test_X[:, 0],
                    'user': test_X[:, 1],
                    'item': test_X[:, 2],
                    'user_nratings': test_X[:, 3],
                    'item_nratings': test_X[:, 4]}

    model_num = 1
    print("models_preds:", models_preds)
    for model in models_preds:
        print("adding model:", model)
        columns_dict_test['model' + str(model_num)] = test_X[:, 4 + model_num]
        model_num += 1

    test_df = pd.DataFrame(columns_dict_test)

    print(test_df)

    FEATURES = list(train_df.columns)
    FEATURES.remove(TARGET)
    print(FEATURES)

    train = train_df
    valid = test_df

    train_features = train[FEATURES]
    valid_features = valid[FEATURES]
    y_train = train[TARGET]

    print('The training set is of length: ', len(train.index))
    print('The test set is of length: ', len(valid.index))

    dtrain = xgb.DMatrix(train_features, label=y_train)
    dvalid = xgb.DMatrix(valid_features)

    params = {
        'base_score': 3.0,
        'eta': 0.03,
        'tree_method': 'exact',
        'max_depth': 2,
        'subsample': 1,
        'colsample_bytree': 0.5,
        'min_child_weight': 1,
        'max_delta_step': 1,
        'eval_metric': 'rmse',
        'objective': 'reg:squarederror',
        'nthread': 8,
        'verbosity': 1,
    }

    meanpredictions = valid[['model1', 'model2', 'model3', 'model4', 'model5', 'model6']].mean(axis=1)
    print("done mean")

    gbm_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=300, verbose_eval=True)
    xgpredictions = gbm_model.predict(dvalid)
    print("done xgb")

    lin_reg = LinearRegression().fit(train_features, y_train)
    linpredictions = lin_reg.predict(valid_features)
    print("done lin")

    ridge = Ridge(alpha=400).fit(train_features, y_train)
    ridgepreds = ridge.predict(valid_features)
    print("done ridge")

    arrays = [np.array(x) for x in [meanpredictions, xgpredictions, linpredictions, ridgepreds]]
    mmpreds = np.clip([np.mean(k) for k in zip(*arrays)], 1.0, 5.0)
    print("done mean mean")
    print(mmpreds)

    with open("ensemblepredictions", "w") as outf:
        for p in mmpreds:
            outf.write(str(p)+"\n")

if __name__ == "__main__":
    main(sys.argv)
