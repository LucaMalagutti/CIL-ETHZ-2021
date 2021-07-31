# Inductive Graph-based Matrix Completion (IGMC)

## Baseline overview

This subdirectory contains a PyTorch (and Geometric-PyTorch) implementation of the [2020 paper "Inductive Matrix Completion Based on Graph Neural Networks"](https://arxiv.org/abs/1904.12058) by Muhan Zhang and Yixin Chen.

The code was originally taken from the [Github](https://github.com/muhanzhang/IGMC) official version released by the authors and adapted to handle the dataset used in the CIL competition.

## Requirements and commands

The code was run using Python 3.7.4. All the required packages are contained in the ``IGMC_requirements.txt`` file. The algorithm uses a high amount of memory, therefore it is highly suggested to run the code on a cluster.

To use the model make sure the directory IGMC/data/CIL, if present, is empty (if there are some folders delete them), then execute ```python Main.py --save-interval 1```.
To generate a submission execute```python generate_submission.py --restore_ckpt results/CIL_valmode/model_checkpointX.pth```, replacing the last "X" with the epoch number of the saved model you want to use to generate the Kaggle submission (the default number of epochs is 5).  

This command eventually generates in IGMC/submissions the file ``IGMC_sub.csv``, which contains a valid submission to be used on the Kaggle website.
