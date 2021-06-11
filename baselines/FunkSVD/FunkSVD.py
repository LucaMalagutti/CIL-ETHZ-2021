from funk_svd.dataset import fetch_ml_ratings
from funk_svd import SVD
from sklearn.metrics import mean_squared_error
import pandas as pd

df = pd.read_csv("../../data/data_train.csv")
df["u_id"] = df['Id'].apply(lambda row: row.split('_')[0][1:])
df["i_id"] = df['Id'].apply(lambda row: row.split('_')[1][1:])
df["rating"] = df["Prediction"]
df = df[['u_id','i_id','rating']]

train = df.sample(frac=0.8, random_state=7)
val = df.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
test = df.drop(train.index.tolist()).drop(val.index.tolist())

svd = SVD(lr=0.0001, 
          reg=0.005, 
          n_epochs=100, 
          n_factors=25, 
          early_stopping=True, 
          shuffle=False,
          min_rating=1, 
          max_rating=5)

svd.fit(X=train, X_val=val)

sub_test = pd.read_csv("../../data/sample_submission.csv")
sub_test["u_id"] = sub_test['Id'].apply(lambda row: row.split('_')[0][1:])
sub_test["i_id"] = sub_test['Id'].apply(lambda row: row.split('_')[1][1:])
sub_test["rating"] = sub_test["Prediction"]
sub_test = sub_test[['u_id','i_id','rating']]

pred = svd.predict(sub_test)
sub_test["Prediction"] = pred
sub_test["Id"] = "r" + sub_test['u_id'] + "_" + "c" + sub_test["i_id"]
sub_test = sub_test[["Id", "Prediction"]]

sub_test.to_csv("test_sub.csv", index=None)




