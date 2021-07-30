# SVD++, Bayesian SVD+ and Bayesian SVD++ Baselines

This subfolder contains the scripts for converting data and training models in LibFM 
(Steffen Rendle (2012): Factorization Machines with libFM, in ACM Trans. Intell. Syst. Technol., 3(3), May. https://www.csie.ntu.edu.tw/~b97053/paper/Factorization%20Machines%20with%20libFM.pdf)

The features for the models are taken from "On the Difficulty of Evaluating Baselines: A Study on Recommender Systems" by Rendle et al. https://arxiv.org/pdf/1905.01395.pdf.  
Bayesian SVD+ corresponds to Bayesian Matrix Factorization in the paper,  
Bayesian SVD++ corresponds to Bayesian SVD++ flipped,  
SVD++ baseline is trained with LibFM using Alternating Least Squares.

## Install LibFM first:

- `git clone https://github.com/srendle/libfm /home/libfm`
- `cd /home/libfm/`
- `make all`
- `export LIBFM_PATH=/home/libfm/bin/`

## Convert data and split train data for validation:

- `cd /.../baselines/BayesianSVD`
- `python CIL_data_converter.py ../../data/data_train.csv`
- `python CIL_data_converter.py ../../data/data_train.csv submission`
- `python CIL_data_converter.py ../../data/sample_submission.csv submission`

## Convert to LibFM format:

- `cd /home/libfm/scripts`
- edit the path to the `baselines` directory in the following commands and execute:
- `./triple_format_to_libfm.pl -in /.../baselines/BayesianSVD/data_split/train90/CIL_data90.train,/.../baselines/BayesianSVD/data_split/valid10/CIL_data10.valid -target 2 -separator "\t"`
- `./triple_format_to_libfm.pl -in /.../baselines/BayesianSVD/data_split/train100/CIL_data100.train,/.../baselines/BayesianSVD/data_split/submission/CIL_data.submission -target 2 -separator "\t"`

## Train Bayesian SVD+ baseline:

- `cd /home/libfm/bin`
- edit the path to the `baselines` directory in the following commands  
- train on 90% and validate on 10%: `./libFM -task r -train /.../baselines/BayesianSVD/data_split/train90/CIL_data90.train.libfm -test /.../baselines/BayesianSVD/data_split/valid10/CIL_data10.valid.libfm -dim 1,1,17 -iter 460 -verbosity 1 `
- train on 100% and generate submission predictions, note that the printed Test score is based on the sample submission, in which every target rating is 3.0: `./libFM -task r -train /.../baselines/BayesianSVD/data_split/train100/CIL_data100.train.libfm -test /.../baselines/BayesianSVD/data_split/submission/CIL_data.submission.libfm -dim 1,1,17 -iter 460 -verbosity 1 -out /.../baselines/BayesianSVD/BayesianSVDPrediction `

## Convert the predictions to the submission format:

- `cd /.../baselines/BayesianSVD`
- `python Libfm_submission_converter.py ../../data/sample_submission.csv BayesianSVDPrediction`

## Generate files for Bayesian SVD++ baseline:

- (if it has not yet been done) convert data and split train data for validation as described above
- `cd /.../baselines/BayesianSVD`
- `python Append_bagofitemsusers.py`  
- `python script_generator.py bagofitemsusers`
- `./scriptbagofitemusers.sh`

## Train Bayesian SVD++ baseline:

- `cd /home/libfm/bin`
- edit the path to the `baselines` directory in the following commands  
- train on 90% and validate on 10%: `./libFM -task r -train /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train90Val10/y/CIL_data90aug.train.libfmy.train -test /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train90Val10/y/CIL_dataaug.valid.libfmy.test -dim 1,1,17 -iter 600 -verbosity 1 --relation /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train90Val10/CIL_data90aug.rel_user,/.../baselines/BayesianSVD/data_features/bagofitemsusers/Train90Val10/CIL_data90aug.rel_item`
- train on 100% and generate submission predictions, note that the printed Test score is based on the sample submission, in which every target rating is 3.0: `./libFM -task r -train /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/y/CIL_data100aug.train.libfmy.train -test /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/y/CIL_dataaug.submission.libfmy.test -dim 1,1,17 -iter 460 -verbosity 1 --relation /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/CIL_data100aug.rel_user,/.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/CIL_data100aug.rel_item -out /.../baselines/BayesianSVD/BayesianSVD++Prediction `

## Convert the predictions to the submission format:

- `cd /.../baselines/BayesianSVD`
- `python Libfm_submission_converter.py ../../data/sample_submission.csv BayesianSVD++Prediction`

## SVD++ baseline:
- `cd /home/libfm/bin`
- train on 100% and generate submission predictions, note that the printed Test score is based on the sample submission, in which every target rating is 3.0:  
  `./libFM -task r -train /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/y/CIL_data100aug.train.libfmy.train -test /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/y/CIL_dataaug.submission.libfmy.test -dim 1,1,17 -verbosity 1 --relation /.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/CIL_data100aug.rel_user,/.../baselines/BayesianSVD/data_features/bagofitemsusers/Train100Submission/CIL_data100aug.rel_item -method als -regular '0,17,35' -iter 93 -seed 5 -out /.../baselines/BayesianSVD/SVD++Prediction`
- convert the predictions to the submission format:
  - `cd /.../baselines/BayesianSVD`
  - `python Libfm_submission_converter.py ../../data/sample_submission.csv SVD++Prediction`
