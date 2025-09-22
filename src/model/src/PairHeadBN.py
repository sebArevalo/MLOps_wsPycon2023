import torch.nn as nn
from . import register_model

class PairHeadBN(nn.Module):
    def __init__(self, input_shape: int = 1026, hidden_layer_1: int = 256, hidden_layer_2: int = 128, num_classes: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, hidden_layer_1),
            nn.BatchNorm1d(hidden_layer_1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_1, hidden_layer_2),
            nn.BatchNorm1d(hidden_layer_2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_2, num_classes),
        )
    def forward(self, x):
        return self.net(x)

@register_model("pairhead_mlp_bn")
def build_pairhead_bn(**cfg):
    return PairHeadBN(**cfg)
