# LightGCN

This subfolder contains a Pytorch re-implementation of the [paper](https://arxiv.org/pdf/2002.02126.pdf) by Xiangnan He et al.
A few parts of this codebase were taken from the [official Pytorch implementation](https://github.com/gusye1234/LightGCN-PyTorch) of the paper.

The vast majority of the code was, however, completely rewritten, to achieve more clarity and to handle the database of the CIL competition correctly. The original prediction task and the model itself were also changed. The original implementation performed Top-N rating prediction using Bayesian Personalized Ranking loss, while our implementation performs a complete user-item rating prediction using a standard RMSE loss. More details on our changed can be found in the Project Report contained in this repo.

## Requirements and commands

This code was tested and executed using Python 3.8.x. The required packages are listed in the file ```LightGCN_requirements.txt```.

To train and evaluate the model run ```python train.py --NAME_OF_YOUR_RUN``` in this subfolder

To generate a submission file, run ```python generate_submission.py --restore_ckpt checkpoints/NAME_OF_SAVED_MODEL```
