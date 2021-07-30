"""
Framework for model selection and hyperparameter tuning of the ensemble, using cross validation
"""

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

SEED = 1
VALID_SIZE = 0.2
TARGET = 'y'

train_features = None
valid_features = None
y_train = None
y_valid = None

train90_file =      "../../baselines/BayesianSVD/data_split/train90/CIL_data90.train"
train100_file =     "../../baselines/BayesianSVD/data_split/train100/CIL_data100.train"
valid_file =        "../../baselines/BayesianSVD/data_split/valid10/CIL_data10.valid"
sub_file =          "../../baselines/BayesianSVD/data_split/submission/CIL_data.submission"

train_out_file = "train_ensemble.csv"
sub_out_file = "sub_ensemble.csv"

def main(args):
    models_preds = args[1:]

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

    FEATURES = list(train_df.columns)
    FEATURES.remove(TARGET)
    print(FEATURES)

    kf = KFold(n_splits=5, random_state=SEED, shuffle=True)
    kf.get_n_splits(train_df)
    print(kf)
    meanscores, linscores, xgscores, ridgescores, mmscores, ridgescores2, ridgescores3, gprscores= [], [], [], [], [], [], [], []
    for train_index, valid_index in kf.split(train_df):
        print("train idx", train_index, "valid idx", valid_index)
        train, valid = train_df.iloc[train_index], train_df.iloc[valid_index]

        train_features = train[FEATURES]
        valid_features = valid[FEATURES]
        y_train = train[TARGET]
        y_valid = valid[TARGET]

        print('The training set is of length: ', len(train.index))
        print('The validation set is of length: ', len(valid.index))

        dtrain = xgb.DMatrix(train_features, label=y_train)
        dvalid = xgb.DMatrix(valid_features, label=y_valid)

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

        meanpredictions = valid[['model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7']].mean(axis=1)
        meanscore = mean_squared_error(y_valid, meanpredictions, squared=False)
        print("\tMean Score {0}\n\n".format(meanscore))
        meanscores.append(meanscore)

        gbm_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=300, verbose_eval=True)
        xgpredictions = gbm_model.predict(dvalid)
                                        # ntree_limit=gbm_model.best_iteration + 1)
        xgscore = mean_squared_error(y_valid, xgpredictions, squared=False)
        print("\tXG Score {0}\n\n".format(xgscore))
        xgscores.append(xgscore)

        lin_reg = LinearRegression().fit(train_features, y_train)
        linpredictions = lin_reg.predict(valid_features)
        linscore = mean_squared_error(y_valid, linpredictions, squared=False)
        print("\tLinreg Score {0}\n\n".format(linscore))
        linscores.append(linscore)

        ridge = Ridge(alpha=400).fit(train_features, y_train)
        ridgepreds = ridge.predict(valid_features)
        ridgescore = mean_squared_error(y_valid, ridgepreds, squared=False)
        print("\tRidgereg Score {0}\n\n".format(ridgescore))
        ridgescores.append(ridgescore)

        ridge2 = Ridge(alpha=300).fit(train_features, y_train)
        ridgepreds2 = ridge2.predict(valid_features)
        ridgescore2 = mean_squared_error(y_valid, ridgepreds2, squared=False)
        print("\tRidgereg2 Score {0}\n\n".format(ridgescore2))
        ridgescores2.append(ridgescore2)

        ridge3 = Ridge(alpha=200).fit(train_features, y_train)
        ridgepreds3 = ridge3.predict(valid_features)
        ridgescore3 = mean_squared_error(y_valid, ridgepreds3, squared=False)
        print("\tRidgereg3 Score {0}\n\n".format(ridgescore3))
        ridgescores3.append(ridgescore3)

        arrays = [np.array(x) for x in [
            meanpredictions,
            xgpredictions,
            linpredictions, ridgepreds, ridgepreds2, ridgepreds3,
        ]]
        mmpreds = np.clip([np.mean(k) for k in zip(*arrays)], 1.0, 5.0)
        mmscore = mean_squared_error(y_valid, mmpreds, squared=False)
        print("\tmm Score {0}\n\n".format(mmscore))
        mmscores.append(mmscore)


    print("Meanscore mean:", np.mean(meanscores), "std:", np.std(meanscores))
    print("linscore mean:", np.mean(linscores), "std:", np.std(linscores))
    print("xgscore mean:", np.mean(xgscores), "std:", np.std(xgscores))
    print("ridgescore mean:", np.mean(ridgescores), "std:", np.std(ridgescores))
    print("ridgescore2 mean:", np.mean(ridgescores2), "std:", np.std(ridgescores2))
    print("ridgescore3 mean:", np.mean(ridgescores3), "std:", np.std(ridgescores3))
    print("mmscore mean:", np.mean(mmscores), "std:", np.std(mmscores))

    return


if __name__ == "__main__":
    main(sys.argv)
