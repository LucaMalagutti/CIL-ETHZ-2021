# Bayesian SVD++ submission

Make sure to have LibFM installed, 
if not follow the instructions in `/baselines/BayesianSVD`

## Empirical Bayes and Scaled weights in Monte Carlo average 

- compile the modified version of LibFM from source:  
  - `cd LibFM_experiments/libfm`
  - `make all`  


- train on 100% and generate submission (replace `/...` with the absolute path to `baselines`):  
  `./bin/libFM -task r -train /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/y/CIL_data100aug.train.libfmy.train -test /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/y/CIL_dataaug.submission.libfmy.test -dim 1,1,17 -iter 570 -verbosity 1 --relation /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/CIL_data100aug.rel_user,/.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/CIL_data100aug.rel_item -out_scaled /.../BayesianSVD++Submission/Ensemble/Emp+ScaledPrediction -T_init 1.0 -T_min 1.0 -alpha_sa 1.0 -scale_init 2.0 -gamma_scale 0.95 -seed 7`


- Convert the predictions to the submission format: 
  - `cd /.../`
  - `python Libfm_submission_converter.py ../../data/sample_submission.csv Emp+ScaledPrediction`


## Ensemble

### Level 1 models

This folder already contains the level 1 model predictions needed for the ensemble.
Here are the instructions to generate the predictions of the level 1 models. 

#### BayesianSVD++ with Empirical Bayes and Scaled weights in Monte Carlo average 
Change the seed and the output name to obtain 5 different pairs of model predictions

- train on 90% and generate predictions for the validation set (change the seed and the output name):  
  `./bin/libFM -task r -train /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train90Val10/y/CIL_data90aug.train.libfmy.train -test /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train90Val10/y/CIL_dataaug.valid.libfmy.test -dim 1,1,17 -iter 570 -verbosity 1 --relation /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train90Val10/CIL_data90aug.rel_user,/.../baselines/BayesianSVD/data_features/bagofitemsusers/Train90Val10/CIL_data90aug.rel_item -out_scaled /.../BayesianSVD++Submission/Ensemble/Emp+ScaledValPrediction -T_init 1.0 -T_min 1.0 -alpha_sa 1.0 -scale_init 2.0 -gamma_scale 0.95 -seed 7`


- train on 100% and generate predictions for the submission set:  
`./bin/libFM -task r -train /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/y/CIL_data100aug.train.libfmy.train -test /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/y/CIL_dataaug.submission.libfmy.test -dim 1,1,17 -iter 570 -verbosity 1 --relation /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/CIL_data100aug.rel_user,/.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/CIL_data100aug.rel_item -out_scaled /.../BayesianSVD++Submission/Ensemble/Emp+ScaledValPrediction -T_init 1.0 -T_min 1.0 -alpha_sa 1.0 -scale_init 2.0 -gamma_scale 0.95 -seed 7`



#### DeepRec
train 1 DeepRec on 90% model and generate predictions for the validation set
and the submission set, following the instructions in `baselines/DeepRec`.

### Level 2 models and final average
- train the level 2 models on the validation predictions of level 1 models
and generate the final predictions:  
 `python ensemble_sub.py /...//BayesianSVD++Submission/Ensemble/ValEmp+ScaledPred1 /...//BayesianSVD++Submission/Ensemble/ValEmp+ScaledPred2 /...//BayesianSVD++Submission/Ensemble/ValEmp+ScaledPred3 /...//BayesianSVD++Submission/Ensemble/ValEmp+ScaledPred4 /...//BayesianSVD++Submission/Ensemble/ValEmp+ScaledPred5 /.../BayesianSVD++Submission/Ensemble/deeprec_predictions_conv /.../BayesianSVD++Submission/Ensemble/Emp+ScaledPred1 /.../BayesianSVD++Submission/Ensemble/Emp+ScaledPred2 /.../BayesianSVD++Submission/Ensemble/Emp+ScaledPred3 /.../BayesianSVD++Submission/Ensemble/Emp+ScaledPred4 /.../BayesianSVD++Submission/Ensemble/Emp+ScaledPred5 /.../BayesianSVD++Submission/Ensemble/deeprec_subpredictions_conv`


- convert the ensemble predictions to the submission format:  
 `python Libfm_submission_converter.py ../../data/sample_submission.csv ensemblepredictions


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
