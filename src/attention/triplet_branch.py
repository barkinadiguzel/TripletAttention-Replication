import torch
import torch.nn as nn

from .z_pool import ZPool
from src.layers.conv_layers import BasicConv


class TripletBranch(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.z_pool = ZPool()
        self.conv = BasicConv(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            relu=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, permute_order=None, inverse_order=None):
        if permute_order is not None:
            x = x.permute(*permute_order).contiguous()

        z = self.z_pool(x)
        a = self.conv(z)
        a = self.sigmoid(a)

        out = x * a

        if inverse_order is not None:
            out = out.permute(*inverse_order).contiguous()

        return out
