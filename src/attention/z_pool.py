import torch
import torch.nn as nn


class ZPool(nn.Module):
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        return torch.cat([max_pool, avg_pool], dim=1)
