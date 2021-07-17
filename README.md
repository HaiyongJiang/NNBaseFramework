This project provides a general framework for deep network based projects. We organized the code so that less efforts are required to work on configuration, training procedures, etc.


## What is needed to use it? 

*) add datasets (nn/data_loader.py).

*) add networks.

*) add evaluators (nn/evaluator.py).

*) add visualization or logging codes (libs/train_callbacks.py)

*) update those parts in (nn/__init__.py)

## TODO

1. complete the evaluation code.
2. add different augmentation for point clouds/ images. 

## Notices: 

This is a very raw version to brew the deep learning based networks, and you need to fill in necessary parts to customize your version.

If you use this project, please keep the license and credit to this page.

## References:

This is an improved version of NNTemplate (https://github.com/HaiyongJiang/NNProjectTemplate).

We reuse some codes from (https://github.com/erikwijmans/Pointnet2_PyTorch.git), e.g. PointNet2, etw_pytorch_utils.
