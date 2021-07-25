# Neural Network Matrix Factorization (NNMF)

## Baseline overview

This subdirectory contains a Tensorflow implementation of the [2015 paper "Neural Network Matrix Factorization"](https://arxiv.org/abs/1511.06443) by Gintare Karolina Dziugaite and Daniel M. Roy.

The code was originally taken from a [Github fork](https://github.com/JoonyoungYi/NNMF-tensorflow) of the original implementation of the paper and adapted to handle the dataset used in the CIL competition.

## Requirements and commands

The code was run using Python 3.8.x. All the required packages are contained in the ``NNMF_requirements.txt`` file

To use the model simply execute ```python run.py```.

This commands starts the training and evaluation processes and eventually generates the file ``NNMF_sub.csv``, which contains a valid submission to be used on the Kaggle website.
