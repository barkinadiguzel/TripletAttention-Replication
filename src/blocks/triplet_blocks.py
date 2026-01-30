import torch
import torch.nn as nn

from src.attention.triplet_attention import TripletAttention


class TripletBlock(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.attention = TripletAttention(kernel_size)

    def forward(self, x):
        return self.attention(x)
