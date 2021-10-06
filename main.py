#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : train.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 17.02.2020
# Last Modified Date: 19.02.2020
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import warnings # should be imported at the top
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from libs import arg_parser
import torch
import torch.optim as optim
import numpy as np
import shutil
import yaml
import glob
import logging
from libs import config
from libs.checkpoints import CheckpointIO
from libs.train_callbacks import (
    TensorboardLoggerCallback, TrainSaverCallback
)
from libs.logger import set_logger


def main(args, gpu_ids=[0]):
    brestore = not args.no_restore
    cfg = config.load_config(args.config, 'configs/default.yaml')
    app_init(cfg, True, brestore)

    out_dir = cfg['training']['out_dir']
    batch_size = cfg['training']['batch_size']*len(gpu_ids)
    batch_size_test = cfg['test']['batch_size']
    bParallel = g_args.ngpu > 1
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda:0" if is_cuda else "cpu")
    layers = ".*"
    if "opt_layers" in cfg["training"]:
        if cfg["training"]["opt_layers"] != ".*":
            layers = cfg["training"]["opt_layers"]
    if args.opt_layers != ".*":
        layers = args.opt_layers

   logging.info("Use device: " + "cuda:0" if is_cuda else "cpu")
    logging.info("Optimizing layers: " + str(layers))

    # Dataset
    train_dataset = config.get_dataset('train', cfg)
    val_dataset = config.get_dataset('val', cfg)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=6, shuffle=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_test, num_workers=2, shuffle=False,
    )

    model = config.get_model(cfg, bParallel, device)
    checkpoint_io = CheckpointIO(os.path.join(out_dir, "checkpoints"), model=model)
    callbacks = [TensorboardLoggerCallback(cfg), TrainSaverCallback(cfg)]
    optimizer, lr_scheduler = config.get_optimizer(cfg, model, layers)
    evaluator = config.get_evaluator(cfg)
    trainer = config.get_trainer(cfg, model, evaluator, device)
    if brestore:
        print("RESTORING###############################################")
        trainer.restore_model(checkpoint_io)
    trainer.print_net_params()

    if args.train:
        logging.info("####################################################")
        logging.info("train the network...")
        trainer.train(train_loader, val_loader, optimizer, lr_scheduler, checkpoint_io, callbacks=callbacks)

    # testing, no tested
    if args.test:
        test_dataset = config.get_dataset('test', cfg)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size_test, num_workers=2, shuffle=False,
        )
        logging.info("####################################################")
        logging.info("test the network...")
        trainer.eval(test_loader, callbacks=[TrainSaverCallback(cfg)])


def app_init(cfg, btrain=True, brestore=True):
    out_dir = cfg['training']['out_dir']
    # logger
    identifier = cfg["model"]["identifier"]
    if not cfg["training"]["out_dir"].endswith(identifier):
        out_dir = os.path.join(
            cfg["training"]["out_dir"], identifier)
        cfg["training"]["out_dir"] = out_dir

    # setup output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    elif not brestore:
        shutil.rmtree(out_dir, ignore_errors=True)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    sub_dirs = ["checkpoints",]
    for fname in sub_dirs:
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            os.makedirs(fpath)

    set_logger(out_dir, cfg["config"]["log_cfg_path"])
    # save configs and source codes
    if btrain:
        cfg_dir = os.path.join(out_dir,"srcs")
        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)
        fname = os.path.join(cfg_dir,"config.yaml")
        with open(fname, 'w') as ofile:
            yaml.dump(cfg, ofile, default_flow_style=False)

        for folder in ["libs", "nn"]:
            dst_dir = os.path.join(cfg_dir, folder)
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(folder, dst_dir)
        shutil.copy("main.py", cfg_dir + "/main.py")


## traing
if __name__ == "__main__":
    parser = arg_parser.get_arg_parser()
    g_args = parser.parse_args()
    g_gpu_ids = arg_parser.setup_GPU(g_args.ngpu)
    main(g_args, g_gpu_ids)

