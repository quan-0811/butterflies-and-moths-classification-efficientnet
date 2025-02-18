import torch
import torch.nn as nn
import torchinfo
from typing import Tuple
from module.squeeze_and_excitation import SqueezeAndExcitation

class MBConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: Tuple[int, int], expansion_ratio: int, se_ratio: float):
        super().__init__()

        self.skip_connection = input_channels == output_channels
        hidden_units = input_channels * expansion_ratio

        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=hidden_units, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.SiLU()
        )
        self.depth_wise = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=kernel_size, stride=1, padding="same", groups=hidden_units),
            nn.BatchNorm2d(num_features=hidden_units)
        )
        self.squeeze_excitation = SqueezeAndExcitation(channels=hidden_units, channel_reduce_ratio=se_ratio)
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=output_channels, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(num_features=output_channels)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.expansion(x)
        x = self.depth_wise(x)
        x = self.squeeze_excitation(x)
        x = self.projection(x)
        if self.skip_connection:
            return x + identity
        else:
            return x


