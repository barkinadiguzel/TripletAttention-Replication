import torch
import torch.nn as nn

from src.blocks.triplet_blocks import TripletAttentionBlock
from src.config import Config


class TripletAttentionModel(nn.Module):
    def __init__(self, in_channels=Config.IN_CHANNELS):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, Config.BASE_CHANNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(Config.BASE_CHANNELS),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.ModuleList([
            TripletAttentionBlock(
                channels=Config.BASE_CHANNELS,
                kernel_size=Config.KERNEL_SIZE
            )
            for _ in range(Config.NUM_BLOCKS)
        ])

        self.head = nn.Conv2d(
            Config.BASE_CHANNELS,
            Config.BASE_CHANNELS,
            kernel_size=1
        )

    def forward(self, x):
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        x = self.head(x)
        return x
