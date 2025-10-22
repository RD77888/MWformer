import torch
import torch.nn as nn
from pytorch_wavelets import DWT1D


class WaveletMultiChannelLoss(nn.Module):
    def __init__(self, wave='db4', level=3, loss_name='L2', delta=1.0,):
        super().__init__()
        self.dwt = DWT1D(wave=wave, J=level, mode='symmetric')
        self.level = level
        self.num_tasks = level + 1
        self.loss_name = loss_name.upper()
        self.delta = delta

        if self.loss_name == "L1":
            self.criterion = nn.L1Loss()
        elif self.loss_name == "L2":
            self.criterion = nn.MSELoss()
        elif self.loss_name == "SMOOTH_L1":
                self.criterion = nn.SmoothL1Loss(beta=delta)

    def forward(self, pred_channels, target):
        if target.dim() == 3 and target.size(2) == 1:
            target = target.transpose(1, 2)
        target = target.unsqueeze(1) if target.dim() == 2 else target
        target_cA, target_cDs = self.dwt(target)
        target_channels = [target_cA] + list(target_cDs)
        indiv_losses = [self.criterion(pred_channels[i], target_channels[i]) for i in range(self.num_tasks)]
        return torch.sum(torch.stack(indiv_losses))

