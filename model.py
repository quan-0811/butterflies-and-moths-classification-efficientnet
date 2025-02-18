import torch
import torch.nn as nn
from module.mbconv import MBConv
from module.global_avg_pooling import GlobalAvgPooling

class EfficientNetB0(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.stage_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.SiLU(),
        )
        self.stage_2 = MBConv(input_channels=32, output_channels=16, kernel_size=(3, 3), stride=1, expansion_ratio=1, se_ratio=0.25)
        self.stage_3 = nn.Sequential(
            MBConv(input_channels=16, output_channels=24, kernel_size=(3, 3), stride=2, expansion_ratio=6, se_ratio=0.25),
            MBConv(input_channels=24, output_channels=24, kernel_size=(3, 3), stride=1, expansion_ratio=6, se_ratio=0.25)
        )
        self.stage_4 = nn.Sequential(
            MBConv(input_channels=24, output_channels=40, kernel_size=(5, 5), stride=2, expansion_ratio=6, se_ratio=0.25),
            MBConv(input_channels=40, output_channels=40, kernel_size=(5, 5), stride=1, expansion_ratio=6, se_ratio=0.25)
        )
        self.stage_5 = nn.Sequential(
            MBConv(input_channels=40, output_channels=80, kernel_size=(3, 3), stride=2, expansion_ratio=6, se_ratio=0.25),
            MBConv(input_channels=80, output_channels=80, kernel_size=(3, 3), stride=1, expansion_ratio=6, se_ratio=0.25),
            MBConv(input_channels=80, output_channels=80, kernel_size=(3, 3), stride=1, expansion_ratio=6, se_ratio=0.25)
        )
        self.stage_6 = nn.Sequential(
            MBConv(input_channels=80, output_channels=112, kernel_size=(5, 5), stride=1, expansion_ratio=6, se_ratio=0.25),
            MBConv(input_channels=112, output_channels=112, kernel_size=(5, 5), stride=1, expansion_ratio=6, se_ratio=0.25),
            MBConv(input_channels=112, output_channels=112, kernel_size=(5, 5), stride=1, expansion_ratio=6, se_ratio=0.25)
        )
        self.stage_7 = nn.Sequential(
            MBConv(input_channels=112, output_channels=192, kernel_size=(5, 5), stride=2, expansion_ratio=6, se_ratio=0.25),
            MBConv(input_channels=192, output_channels=192, kernel_size=(5, 5), stride=1, expansion_ratio=6, se_ratio=0.25),
            MBConv(input_channels=192, output_channels=192, kernel_size=(5, 5), stride=1, expansion_ratio=6, se_ratio=0.25),
            MBConv(input_channels=192, output_channels=192, kernel_size=(5, 5), stride=1, expansion_ratio=6, se_ratio=0.25)
        )
        self.stage_8 = MBConv(input_channels=192, output_channels=320, kernel_size=(3, 3), stride=1, expansion_ratio=6, se_ratio=0.25)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(num_features=1280),
            nn.SiLU(),
            GlobalAvgPooling(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=1280, out_features=out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.stage_5(x)
        x = self.stage_6(x)
        x = self.stage_7(x)
        x = self.stage_8(x)
        x = self.classifier(x)
        return x




