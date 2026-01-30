import torch
import torch.nn as nn

from .triplet_branch import TripletBranch


class TripletAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.branch_cw = TripletBranch(kernel_size)
        self.branch_ch = TripletBranch(kernel_size)
        self.branch_hw = TripletBranch(kernel_size)

    def forward(self, x):
        # C-W branch
        out_cw = self.branch_cw(
            x,
            permute_order=(0, 2, 1, 3),
            inverse_order=(0, 2, 1, 3),
        )

        # C-H branch
        out_ch = self.branch_ch(
            x,
            permute_order=(0, 3, 2, 1),
            inverse_order=(0, 3, 2, 1),
        )

        # H-W branch 
        out_hw = self.branch_hw(x)

        out = (out_cw + out_ch + out_hw) / 3.0
        return out
