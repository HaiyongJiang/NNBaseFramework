from nn.simple_net import SimpleNet
from nn.data_loader import DatasetBase
from nn.evaluator import EvalBase

dataset_dict = {
    "DatasetBase": DatasetBase,
}
method_dict = {
    "SimpleNet": SimpleNet,
}
eval_dict = {
    "EvalBase": EvalBase
}


