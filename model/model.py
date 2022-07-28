import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class EffNet(nn.Module):
    def __init__(self, num_classes=5):
        super(EffNet, self).__init__()
        self.eff = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes, in_channels=1)
    def forward(self, x):
        x = self.eff(x)
        return x