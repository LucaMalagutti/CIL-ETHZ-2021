# Deep Autoencoders for Collaborative Filtering

This subfolder contains an implementation of the [paper](https://arxiv.org/pdf/1708.01715.pdf) "Training Deep Autoencoders for Collaborative Filtering" by Oleksii Kuchaiev and Boris Ginsburg.

This code was taken from the [official Github repo](https://github.com/NVIDIA/DeepRecommender) of the paper.

## Requirements

All the requirements needed to reproduce the baseline results are contained in the file ``DeepRec_requirements.txt``
This code was tested when using Python 3.8.x

## Commands
All the following commands are to be run when inside this (```baselines/DeepRec/```) subfolder.

To train and validate the model run the command

```python run.py --dropout=0.4 --layer1_dim=512 --layer2_dim=32 --layer3_dim=64 --learning_rate=0.0035 --weight_decay=4.33e-08```

To generate a submission after training, run

```python infer.py --path_to_train_data data/train90 --path_to_eval_data data/submission --hidden_layers 512,32,64 --save_path model_save/model.epoch_X```

replacing the last "X" with the epoch number of the saved model you want to use to generate the Kaggle submission.

## Dataset generation

This subfolder already contains all the data necessary to train, evaluate and test the model in the ``/data`` folder.

All the data files can, however, be generated again using the following commands.

To generate the training and the validation dataset run

```python data_utils/CIL_data_converter.py ../../data/data_train.csv```

To generate the test (submission) dataset run

```python data_utils/CIL_data_converter.py ../../data/sample_submission.csv submission```
