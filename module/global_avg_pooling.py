import torch
import torch.nn as nn
from typing import Tuple

class GlobalAvgPooling(nn.Module):
    def __init__(self, output_size: Tuple[int, int]):
        super().__init__()
        self.output_size = output_size
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H_in, W_in = x.shape
        H_out, W_out = self.output_size
        avg_pool_layer = nn.AvgPool2d(kernel_size=(H_in // H_out, W_in // W_out))
        return avg_pool_layer(x)