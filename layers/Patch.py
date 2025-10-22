import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RMSnorm import RMSNorm


class TFPE(nn.Module):
    def __init__(self, target_patch_num, band_len, input_channels, d_model):
        super().__init__()
        self.input_channels = input_channels
        self.target_patch_num = target_patch_num

        self.patch_len = math.ceil(band_len / target_patch_num)
        if self.patch_len == 0:
            self.patch_len = 1
        self.actual_patch_num = target_patch_num
        self.padded_total_len = self.patch_len * self.actual_patch_num

        self.relative_time_embedder = nn.Embedding(self.patch_len, d_model // 8)
        self.feature_extractor = TimeAwareFeatureExtractor(
            input_channels, d_model, self.patch_len
        )

        self.pos_encoding = nn.Parameter(
                torch.randn(1, self.actual_patch_num, d_model)
            )
        self.norm_output = RMSNorm(d_model)

    def forward(self, x):
        batch, current_channels, current_band_len = x.shape

        x = self._handle_padding(x, current_band_len)
        x = self.feature_extractor(x, self.relative_time_embedder)

        x = x.permute(0, 2, 1)  # [B, num_patches, d_model]

        x = x + self.pos_encoding
        x = self.norm_output(x)
        return x

    def _handle_padding(self, x, current_len):
        total_pad = max(self.padded_total_len - current_len, 0)
        if total_pad > 0:
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            x = F.pad(x, (pad_left, pad_right), mode='replicate')
        elif current_len > self.padded_total_len:
            start = (current_len - self.padded_total_len) // 2
            x = x[..., start:start + self.padded_total_len]
        return x


class TimeAwareFeatureExtractor(nn.Module):
    def __init__(self, input_channels, d_model, patch_len):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.conv1 = nn.Conv1d(input_channels, d_model // 2,
                               kernel_size=3, padding=1, bias=False)
        self.time_gate = nn.Sequential(
            nn.Linear(d_model // 8, d_model // 2),
            nn.Sigmoid()
        )
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(d_model // 2, d_model,
                               kernel_size=patch_len,
                               stride=patch_len,
                               bias=False)

    def forward(self, x, time_embedder):
        B, C, L = x.shape
        num_patches = L // self.patch_len

        x = self.conv1(x)  # [B, d_model//2, L]
        time_positions = torch.arange(self.patch_len, device=x.device)
        time_positions = time_positions.repeat(num_patches)[:L]
        time_embeds = time_embedder(time_positions)  # [L, d_model//8]

        gate_weights = self.time_gate(time_embeds)  # [L, d_model//2]
        gate_weights = gate_weights.T.unsqueeze(0)  # [1, d_model//2, L]

        x = x * gate_weights
        x = self.activation(x)

        x = self.conv2(x)
        return x
