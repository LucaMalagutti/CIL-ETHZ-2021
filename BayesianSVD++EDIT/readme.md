# Bayesian SVD++ experiments

Make sure to have LibFM installed, 
if not follow the instructions in `/baselines/BayesianSVD`

## DeepRec features

- (if it has not yet been done) convert and split data for DeepRec:
  - `cd baselines/DeepRec`
  - `python data_utils/CIL_data_converter.py ../../data/data_train.csv`
  - `python data_utils/CIL_data_converter.py ../../data/data_train.csv submission`
  - `python data_utils/CIL_data_converter.py ../../data/sample_submission.csv submission`
  

- extract deep features for every user in the train and validation split, 
  using DeepRec trained only on the train split:  
  - `cd ../../BayesianSVD++EDIT/DeepRec_features`
  - `python ../../baselines/DeepRec/infer_deepfeatures.py --path_to_train_data ../../baselines/DeepRec/data/train90 --path_to_eval_data ../../baselines/DeepRec/data/train90 --hidden_layers 128,32,128 --non_linearity_type selu --save_path model_save/modelVal0.9809.epoch_84 --drop_prob 0.0 --predictions_path deepfeatures90_2ndBottleneck.json`


- by default deep features are extracted from the representation in the 
  2nd bottleneck in DeepRec, to extract deep features from the 2nd bottleneck in
  `baselines/DeepRec/infer_deepfeatures.py` change the argument passed to 
  `rencoder = model.AutoEncoder()` from `deepf_layer="second_bottleneck"` to
  `deepf_layer="first_bottleneck"`
  

- extract deep features for every user in the full training data and
  submission data:  
  `python ../../baselines/DeepRec/infer_deepfeatures.py --path_to_train_data ../../baselines/DeepRec/data/train100 --path_to_eval_data ../../baselines/DeepRec/data/train100 --hidden_layers 128,32,128 --non_linearity_type selu --save_path model_save/modelVal0.9809.epoch_84 --drop_prob 0.0 --predictions_path deepfeatures100_2ndBottleneck.json`
 
 
- generate train, validation, submission files for LibFM containing implicit
  user and item information (Bayesian SVD++) and deep features:  
  `python Append_deepf_bags.py`

  
- generate the script to convert files to LibFM format:  
  `python script_generator.py deepf_bags`
  

- the script assumes that LibFM is installed in
  `/home/libfm`, edit the first line of the script if
  it is different. Then run the script 
  (add `sudo bash` if necessary):  
  `./scriptdeepf_bags.sh`
  

- change the path to `data_features` in the following commands 
  (substitute `/.../` with the absolute path)


- train LibFM on 90% of the data and validate on 10%:
  - `cd /home/libfm/bin`
  - `./libFM -task r -train /.../data_features/Train90Val10/y/CIL_data90aug.train.libfmy.train -test /.../data_features/Train90Val10/y/CIL_dataaug.valid.libfmy.test -dim 1,1,17 -iter 460 -verbosity 1 --relation /.../data_features/Train90Val10/CIL_data90aug.rel_user,/.../data_features/Train90Val10/CIL_data90aug.rel_item --meta /.../data_features/Train90Val10/CIL_data90aug.rel_user.groups`
  

- train on 100% of the data and generate submission:
  `./libFM -task r -train /.../data_features/Train100Submission/y/CIL_data100aug.train.libfmy.train -test /.../data_features/Train100Submission/y/CIL_dataaug.submission.libfmy.test -dim 1,1,17 -iter 460 -verbosity 1 --relation /.../data_features/Train100Submission/CIL_data100aug.rel_user,/.../data_features/Train100Submission/CIL_data100aug.rel_item -out /.../data_features/../BayesianSVD++DeepfPrediction`
  

- Convert the predictions to the submission format: 
  - `cd /.../`
  - `python Libfm_submission_converter.py ../../data/sample_submission.csv BayesianSVD++DeepfPrediction`

