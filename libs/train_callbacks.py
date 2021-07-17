#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : train_callbacks.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 19.08.2018
# Last Modified Date: 19.02.2020
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import os
import torch
from tensorboardX import SummaryWriter
import numpy as np
import libs.visualize as vis
from libs.io import export_pointcloud

class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def add_dict(self, writer, val_dict, prefix, epoch_id, val_type="scalar"):
        for k,v in val_dict.items():
            if val_type=="scalar":
                if isinstance(v, (int, float)):
                    writer.add_scalar(prefix+"/"+k, v, epoch_id)
            elif val_type=="hist":
                writer.add_histogram(prefix+"/"+k, v, epoch_id)


class TensorboardLoggerCallback(Callback):
    def __init__(self, cfg):
        """
            Callback intended to be executed at each epoch
            of the training which goal is to add valuable
            information to the tensorboard logs such as the losses
            and accuracies
        Args:
            path_to_files (str): The path where to store the log files
        """
        BASE_DIR = cfg["training"]["out_dir"]
        self.path_to_files = os.path.join(BASE_DIR, "logs")
        if not os.path.exists(self.path_to_files):
            os.makedirs(self.path_to_files)
        self.loss = 1e10

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != "epoch":
            return
        epoch_id = kwargs['epoch_id']

        ## add loss for tensorboard visualization
        self.writer = SummaryWriter(self.path_to_files)
        for name in ["train_loss", "val_loss"]:
            self.add_dict(self.writer, kwargs[name], 'data/%s'%(name), epoch_id)
        self.add_dict(self.writer, {"lr": kwargs['lr']}, 'data/learning_rate', epoch_id)
        lr = kwargs['lr']
#        if "train_sample" in kwargs and epoch_id>0:
#            sample = kwargs["train_sample"]
#            for k in sample:
#                if k.endswith("weight"):
#                    if "param_" in k:
#                        self.add_dict(self.writer, {k:lr*sample[k]}, 'param/', epoch_id, "hist")
#                    if "update_" in k:
#                        self.add_dict(self.writer, {k:lr*sample[k]}, 'update/', epoch_id, "hist")
        self.writer.close()


class TrainSaverCallback:
    def __init__(self, cfg):
        self.cfg = cfg
        self.log_interval = cfg["training"]["visualize_every"]
        self.max2save = cfg["training"]["saver_max_num"]
        BASE_DIR = cfg["training"]["out_dir"]

        self.im_path = os.path.join(BASE_DIR, "images")
        self.shape_path = os.path.join(BASE_DIR, "results")
        for f in [self.im_path, self.shape_path]:
            if not os.path.exists(f):
                os.makedirs(f)

    def __call__(self, *args, **kwargs):
        """ Save Input/Target/Predict/Diff, Corner/Line/Poly_maps """
        if kwargs['step_name'] != "epoch":
            return
        epoch = 1
        if 'epoch_id' in kwargs:
            epoch = kwargs['epoch_id']
        if epoch % self.log_interval!= 0:
            return

        for split in ["val", "test"]:
            if split+"_sample" not in kwargs \
                    or len(kwargs[split + "_sample"]) == 0:
                continue
            continue
