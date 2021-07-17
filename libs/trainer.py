#!/usr/bin/env python
# -*- coding: u8 -*-
# File              : net_optim.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 19.08.2018
# Last Modified Date: 19.02.2020
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import torch
import numpy as np
import GPUtil
import logging
import time

class Trainer:
    """
    Network optimizer/trainer.
    """
    def __init__(self, cfg, net, evaluator, device):
        self.cfg = cfg
        self.net = net
        self.evaluator = evaluator
        self.device = device

        self.epoch_it = 0
        self.max_epochs = cfg["training"]["max_epochs"]
        self.lr = self.cfg["training"]["lr"]
        self.loss_val_best = np.inf

    def print_net_params(self):
        logging.info("Network architecture: ")
        logging.info(self.net)
        nparameters = sum(p.numel() for p in self.net.parameters())
        logging.info('Total number of parameters: %d' % nparameters)

    def restore_model(self, ckp, brestart = False):
        """
        ckp: checkpoint_io
        brestart (bool): if we set lr based on restored model
        """
        model_path = self.cfg["model"]["ckp_path"]
        if not os.path.exists(model_path):
            logging.error("The specified model path does not exist: " + model_path)
            return
        try:
            load_dict = ck.load(model_path)
        except FileExistsError:
            load_dict = dict()
        if not brestart:
            self.epoch_it = load_dict.get('epoch_it', -1)
            self.loss_val_best = load_dict.get('loss_val_best', np.inf)
        if "model" in load_dict:
            self.net.load_state_dict(load_dict["model"])
        logging.info("Restore model trained with epoch=%d, lr=%f, loss=%f"%(
            self.epoch_it, self.lr, self.loss_val_best)
        )

    def _update_metrics(self, metrics, loss, metric):
        """
        Update given losses and metrics. The update is in-place

        params:
            @metrics: a dict, the summary of metrics
            @loss: a dict, the present measured loss
            @metric, a dict or list the present measured metric.
        """
        for k,v in loss.items():
            metrics.setdefault(k, 0)
            metrics[k] = metrics[k] + loss[k]

        assert(isinstance(metric, dict))
        for k,v in metric.items():
            flag = k in metrics
            metrics.setdefault(k, v)

            if isinstance(v, dict):
                for k1 in v:
                    if k1 not in metrics[k]:
                        metrics[k][k1] = metric[k][k1]
                    else:
                        metrics[k][k1] += metric[k][k1]
            elif "_min" in k:
                metrics[k] = min(metric[k], metrics[k])
            elif "_max" in k:
                metrics[k] = max(metric[k], metrics[k])
            elif "_avg" in k or "_sum" in k:
                if flag:
                    metrics[k] = metric[k] + metrics[k]
            else:
                if flag:
                    metrics[k] = metric[k] + metrics[k]

        return metrics

    def _normalize_metrics(self, metrics, count):
        """
        Normalize the items in metrics with a size, nsample. The normalization is in-place.
        """
        for k in metrics:
            if "_max" in k or "_min" in k:
                metrics_ret[k] = metrics[k]
            elif "_avg" in k or k == "loss_sum":
                metrics[k] /= count
            else: ## average in default
                metrics[k] /= count
        return metrics

    def get_largest_lr(self, opt):
        return np.max([v["lr"] for v in opt.param_groups])

    def eval(self, test_loader, callbacks=None):
        """
            Trains the neural net
        Args:
            valid_loader (DataLoader): The Dataloader for testing
            callbacks (list): List of callbacks functions to call at each epoch
        Returns:
            str, None: The path where the model was saved, or None if it wasn't saved
        """
        t_start = time.time()
        with torch.no_grad():
            test_metric, test_sample = self._validate_epoch(test_loader)
        t_elapsed = time.time() - t_start

        if callbacks:
            for cb in callbacks:
                cb(step_name="epoch", net=self.net,
                epoch_id=0, lr = 0.0,
                train_sample=[], val_sample=test_sample,
                train_loss= {}, val_loss= test_metric,
                )

        logging.info("/***********************************************/")
        logging.info("Testing loss: /n" + str(test_metric) )
        logging.info("Average timing on %d batches: %fs" % (
            len(test_loader), t_elapsed/len(test_loader)) )

    def train(self, train_loader, valid_loader, optimizer,
              lr_scheduler, checkpoint_io, callbacks=None, **kwargs):
        """
            Trains the neural net
        Args:
            train_loader (DataLoader): The Dataloader for training
            valid_loader (DataLoader): The Dataloader for validation
            callbacks (list): List of callbacks functions to call at each epoch
        Returns:
            str, None: The path where the model was saved, or None if it wasn't saved
        """
        # restore lr
        for ii in range(self.epoch_it):
            if self.cfg["training"]["scheduler"] is None:
                pass
            elif self.cfg["training"]["scheduler"]=="ReduceLROnPlateau":
                lr_scheduler.step(val_loss["loss_sum"])
            else:
                lr_scheduler.step()
        lr = self.get_largest_lr(optimizer)
        states = {"lr": lr, "state_dict": [], "val_loss": self.loss_val_best}
        logging.info("Start with a learning rate of " + str(lr))

        identifier = self.cfg["model"]["identifier"]
        for epoch in range(self.epoch_it+1, self.max_epochs):
            self.epoch_it = epoch
            logging.info("\n\nEpoch#%d, learning rate#%e, model: %s"%(
                epoch, lr, identifier) )
            train_loss, train_sample = self._train_epoch(
                train_loader, optimizer, epoch)

            # Run the validation pass
            val_loss, val_sample = {}, {}
            if epoch%self.cfg["training"]["validate_every"] == 0:
                with torch.no_grad():
                    val_loss, val_sample = self._validate_epoch(valid_loader)

                if states["val_loss"] >= -val_loss["total_metric"]:
                    logging.info("Update best model: %.3f, %.3f"%(
                        -val_loss["total_metric"], states["val_loss"]))
                    states["val_loss"] = -val_loss["total_metric"]
                    states["val_metric"] = val_loss
                    states["epoch"] = epoch
                    checkpoint_io.save('model_best.pt', epoch_it=epoch,
                                    loss_val_best=states["val_loss"])

            checkpoint_io.save('model.pt', epoch_it=epoch,
                            loss_val_best=states["val_loss"])

            # Reduce learning rate if needed
            if lr_scheduler == None:
                pass
            elif self.cfg["training"]["scheduler"]=="ReduceLROnPlateau":
                lr_scheduler.step(val_loss["loss_sum"])
            else:
                lr_scheduler.step()
            lr = self.get_largest_lr(optimizer)

            if callbacks:
                for cb in callbacks:
                    cb(step_name="epoch",
                    net=self.net,
                    train_sample=train_sample,
                    val_sample=val_sample,
                    epoch_id=epoch,
                    train_loss= train_loss,
                    val_loss=val_loss,
                    lr = lr,
                    )
            if lr <= 1e-7:
                break

        if "epoch" in states:
            logging.info("Best validation results at epoch#%d:"%states["epoch"])
            logging.info("learning rate: %f"%states["lr"])
            logging.info("val loss: " + str(states["val_metric"]))

    def _train_epoch(self, train_loader, optimizer, epoch):
        """
        returns:
            @metrics: evaluation metrics and loss terms
            @samples: a batch of examples
        """
        self.net.train()
        metrics = {}
        loss_names = {}
        batch_idx = 0
        for batch_idx, data in enumerate(train_loader):
            data = self._to_device(data)
            samples, loss, metric = self.train_step(data, optimizer)
            self._update_metrics(metrics, loss, metric)

            if batch_idx % self.cfg["training"]["print_per_batch"] == 0:
                logging.info('Train Epoch: {} [{}*({:.0f}%), {}]'.format(
                    epoch, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), batch_idx
                ))
                logging.info("Loss terms: " + ", ".join(["%s: %.4f"%(k,loss[k])
                                           for k in sorted(loss.keys())]))
        batch_idx += 1
        metrics = self._normalize_metrics(metrics, batch_idx)
        logging.info("Train set: ")
        logging.info("Loss terms: " + ", ".join([
            "%s: %.4f"%(k, metrics[k]) for k in sorted(loss.keys())]))
        logging.info('Metrics: ' + ", ".join([
            "%s: %.4f"%(k, metrics[k]) for k in sorted(metrics.keys())
            if k not in loss
        ]))
        return metrics, samples

    def _validate_epoch(self, val_loader, btest=False):
        self.net.eval()
        sample_list = {}
        loss = {}
        metrics = {}
        for batch_idx, data in enumerate(val_loader):
            data = self._to_device(data)
            samples, loss, metric = self.eval_step(data)
            self._update_metrics(metrics, loss, metric)
            if batch_idx == 0:
                sample_list = {k:samples[k].detach() for k in samples}
            elif batch_idx<20:
                sample_list = {
                    k: torch.cat([sample_list[k], samples[k].detach()], dim=0)
                    for k in samples
                }

        batch_idx += 1
        metrics = self._normalize_metrics(metrics, batch_idx)
        logging.info("Validation set: ")
        logging.info("Loss terms: " + ", ".join([
            "%s: %.4f"%(k, metrics[k]) for k in sorted(loss.keys())]))
        logging.info('Metrics: ' + ", ".join([
            "%s: %.4f"%(k, metrics[k]) for k in sorted(metrics.keys())
            if k not in loss
        ]))
        return metrics, sample_list


    ## ////////////////////////////////////////////////////////////
    ## debugs
    def _register_params(self, sample):
        state_dict = self.net.state_dict(keep_vars=True)
        for k in state_dict:
            if k.endswith("weight"):
                sample["param_"+k] = state_dict[k].data.cpu().numpy()
                sample["grad_"+k] = state_dict[k].grad.data.cpu().numpy()
                sample["update_"+k] = np.abs(sample["grad_"+k])/(1e-8+np.abs(sample["param_"+k]))


    def train_step(self, data, optimizer):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.net.train()
        optimizer.zero_grad()
        preds = self.net(data)
        samples, loss_dict, metrics = self.evaluator.apply(data, preds)
        loss_dict["loss_sum"].backward()
        optimizer.step()
        losses = {k:loss_dict[k].item() for k in loss_dict}

        return samples, losses, metrics

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.net.eval()
        with torch.no_grad():
            preds = self.net(data)
            samples, loss_dict, metrics = self.evaluator.apply(data, preds)
        losses = {k:loss_dict[k].item() for k in loss_dict}
        return samples, losses, metrics

    def _to_device(self, data):
        '''Place data on a device, where data is stored as a dict.
        '''
        for k in data:
            data[k] = data[k].to(self.device)
        return data

