import torch
import torch.nn as nn

class EvalBase:
    """
    Network evaluator

    WARNING: you should avoid gradient propagation in compute_metric and get_samples.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss_weights = cfg["training"]["losses"]

    def apply(self, data, preds):
        samples = self._get_samples(data, preds)
        loss = self._compute_loss(data, preds)
        metric = self._compute_metric(data, preds)
        return samples, loss, metric

    def _compute_loss(self, data, preds):
        """
        The objective functions to be optimized.

        WARNING: gradients required, loss_sum required.
        """
        lb_gnd = data["output"]
        logits = preds.transpose(1,2)
        loss = nn.CrossEntropyLoss()(logits, lb_gnd)
        return {"loss_sum": loss}

    def _compute_metric(self, data, preds):
        """
        The metrics for evaluation.

        WARNING: no gradients, total_metric required.
        """
        raise Exception("Error: overriding required.")
        return {"acc": 0.0}

    def _get_samples(self, data, preds):
        """
        Get samples for visualization.

        WARNING: no gradients.
        """
        k = 10
        return {"input": data["input"][:, :k].detach(),
                "out": preds[:, :k].detach()}

