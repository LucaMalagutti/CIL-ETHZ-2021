# DeepRec++ - Jointly train Autoencoders and MLP for collaborative filtering

The idea behind this model consists in training an MLP to predict the dataset's ratings, the inputs to the model are the internal vector representations of users and items extracted from pretrained autoencoders. We pretrained autoencoders designed in [paper](https://arxiv.org/pdf/1708.01715.pdf) "Training Deep Autoencoders for Collaborative Filtering" by Oleksii Kuchaiev and Boris Ginsburg making use of the code available in the [official Github repo](https://github.com/NVIDIA/DeepRecommender) of the paper.

Our model can be trained jointly, finetuning the autoencoders along with the MLP or using fixed pretrained autoencoder representations for both items and users.

## Requirements

All the requirements needed to reproduce the baseline results are contained in the file ``DeepRec++_requirements.txt``
This code was tested when using Python 3.8.x

## Dataset generation

All the data files need to train and evaluate the model can be generated using the following commands.

To generate the training and the validation dataset run

```python data_utils/CIL_data_converter.py ../data/data_train.csv```

To generate the test (submission) dataset run

```python data_utils/CIL_data_converter.py ../data/sample_submission.csv submission```

## Commands
All the following commands are to be run when inside this (```DeepRec++/```) subfolder.

To train and validate the model with joint training of the MLP and the autoencoders, run

```python run_alter.py --joint_train --pretrain_autoencoders```

To train and validate the model with separate training of the MLP and the autoencoders, run

```source run_separate.sh```

To generate a submission after joint training, run

```source infer_alter.sh```

To generate a submission after separate training, run

```python infer_mlp.py --logdir=model_save/mlp/model.last --input_size=576```
