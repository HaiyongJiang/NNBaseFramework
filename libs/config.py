#!/usr/bin/env python3
# -*- coding: u8 -*-
# File              : config.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 26.01.2020
# Last Modified Date: 20.02.2020
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import yaml
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import *
from torchvision import transforms
from nn import dataset_dict, method_dict, eval_dict
from libs.trainer import Trainer
import re
import logging


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (str): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, yaml.FullLoader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, yaml.FullLoader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            if dict1[k] is None and v is not None:
               dict1[k] = dict()
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# index network models
def get_model(cfg, bParallel=True, device="cuda"):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        bParallel (bool): use gpu parallel or not
        device (str): pytorch device
    '''
    method_name = cfg['method']
    net = method_dict[method_name](cfg)
    if bParallel:
        net = nn.DataParallel(net)
    return net.to(device)


# Trainer
def get_trainer(cfg, model, evaluator, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
    '''
    return Trainer(cfg, model, evaluator, device)

# Evaluator
def get_evaluator(cfg):
    ''' Returns evaluator.

    Args:
        cfg (dict): config dictionary
    '''
    eval_name = cfg['model']['eval_method']
    return eval_dict[eval_name](cfg)

# index datasets
def get_dataset(mode, cfg):
    ''' Returns the dataset.

    Args:
        mode (enum) : train/val/test
        cfg (dict): config dictionary
    '''
    dataset_name = cfg['data']['dataset']
    if dataset_name not in dataset_dict:
        logging.warning("Available datasets: %s" % (",".join(dataset_dict.keys())))
        raise Exception("Error dataset name.")
    return dataset_dict[dataset_name](mode, cfg)


def get_params(model, pattern):
    ''' Returns parameters matching the regex pattern.

    Args:
        model (class) : the network
        pattern (str) : the regex pattern
    '''
    params = model.named_parameters()
    params = {k:v for k,v in params if re.match(pattern, k) is not None}
    logging.info("Filter parameters with regex (%s): " % pattern)
    logging.info(",".join(params.keys()))
    return params.values()


def get_optimizer(cfg, model, patterns =".*"):
    ''' Returns an optimizer with proper lrs.

    Args:
        cfg (dict): configurations
        model (class) : the network
        pattern (str/dict) : a regex pattern or a dict or regex patterns
    '''
    optimizer_func = None
    if cfg["training"]["optimizer"] == "ADAM":
        optimizer_func = lambda x,y: optim.Adam(x, y)
    elif cfg["training"]["optimizer"] == "ADAMW":
        optimizer_func = lambda x,y: optim.AdamW(x, y)
    elif cfg["training"]["optimizer"] == "SGD":
        optimizer_func = lambda x,y: optim.SGD(x, y, momentum=0.9)
    else:
        raise "Unexpected optimizer: " + cfg["training"]["optimizer"]

    ## optimizer
    lr = float(cfg["training"]["lr"])
    if isinstance(patterns, str):
        optimizer = optimizer_func(get_params(model, patterns), lr)
    elif isinstance(patterns, dict):
        param_list = []
        for name, lr in patterns.items():
            param_list.append({"params": get_params(model, name), "lr": lr})
        optimizer = optimizer_func(param_list, lr)

    ## scheduler
    lr_scheduler = None
    scheduler_name = cfg["training"]["scheduler"]
    scheduler_params = cfg["training"]["scheduler_params"]
    logging.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    logging.info("scheduler: " + str(scheduler_name))
    logging.info("params: " + str(scheduler_params))
    logging.info("optimizer: " + cfg["training"]["optimizer"])
    logging.info("init lr = " + str(cfg["training"]["lr"]))
    logging.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    if scheduler_name == "ReduceLROnPlateau":
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', **scheduler_params)
    elif scheduler_name == "StepLR":
        lr_scheduler = StepLR(optimizer,
                                scheduler_params["step_size"],
                                gamma=scheduler_params["gamma"]
                                )
    elif scheduler_name == "MultiStepLR":
        lr_scheduler = MultiStepLR(optimizer, **scheduler_params)
    return optimizer, lr_scheduler
