import torch
import torch.nn as nn
from typing import Tuple

class MBConv(nn.Module):
    def __init__(self, input_channels: int, kernel_size: Tuple[int, int], expansion_factor: int):
        super().__init__()
        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels*expansion_factor, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(num_features=input_channels*expansion_factor),
            nn.SiLU()
        )