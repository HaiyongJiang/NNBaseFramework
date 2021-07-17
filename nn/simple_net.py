import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    r"""
    This an example network for framework testing
    """
    def  __init__(self, cfg):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, data):
        inp = data["input"]
        return self.fc(inp)
