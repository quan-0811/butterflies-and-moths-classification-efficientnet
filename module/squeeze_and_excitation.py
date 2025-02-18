import torch
import torch.nn as nn
from module.global_avg_pooling import GlobalAvgPooling

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels: int, channel_reduce_ratio: int):
        super().__init__()
        self.squeeze = GlobalAvgPooling(output_size=(1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // channel_reduce_ratio),
            nn.SiLU(),
            nn.Linear(in_features=channels // channel_reduce_ratio, out_features=channels),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = self.squeeze(x)
        excitation = self.excitation(squeeze.squeeze())
        return x * excitation.unsqueeze(-1).unsqueeze(-1)


