import torch
import torch.nn as nn
from module.global_avg_pooling import GlobalAvgPooling

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels: int, channel_reduce_ratio: float):
        super().__init__()

        squeeze_units = int(channels * channel_reduce_ratio)

        self.squeeze = GlobalAvgPooling(output_size=(1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_features=channels, out_features=squeeze_units),
            nn.SiLU(),
            nn.Linear(in_features=squeeze_units, out_features=channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = self.squeeze(x)
        excitation = self.excitation(squeeze.squeeze())
        return x * excitation.unsqueeze(-1).unsqueeze(-1)


