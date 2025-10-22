import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6)
        return self.scale * x / rms
