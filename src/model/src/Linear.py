import torch.nn as nn
from . import register_model

class LinearHead(nn.Module):
    def __init__(self, input_shape: int = 1026, num_classes: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_shape, num_classes)
    def forward(self, x):
        return self.fc(x)

@register_model("linear")
def build_linear(**cfg):
    return LinearHead(**cfg)
