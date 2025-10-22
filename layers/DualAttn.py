import torch
import torch.nn as nn
from einops import rearrange

from layers.Attn import MultiHeadAttention
from layers.RMSnorm import RMSNorm


class LearnableFusion(nn.Module):
    def __init__(self, num_branches=2, d_model=256):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_branches),
            nn.Softmax(dim=-1)
        )

    def forward(self, *xs):
        b, _, _, d = xs[0].shape
        avg_feat = torch.mean(xs[0], dim=(1, 2))
        weights = self.fusion_mlp(avg_feat)
        return sum(
            w.unsqueeze(1).unsqueeze(2).unsqueeze(3) * x
            for w, x in zip(weights.unbind(dim=-1), xs)
        )

class TemporalAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model,d_model, n_heads, dropout=dropout)
    def forward(self, x):
        # x: [b, n, f, d]
        x_reshaped = rearrange(x, 'b n f d -> (b f) n d')  # [b*f, n, d]
        out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)
        out = rearrange(out, '(b f) n d -> b n f d', b=x.shape[0])  # [b, n, f, d]
        return out

class FreqAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model,d_model, n_heads, dropout=dropout)
    def forward(self, x):
        # x: [b, n, f, d]
        x_reshaped = rearrange(x, 'b n f d -> (b n) f d')  # [b*n, f, d]
        out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)
        out = rearrange(out, '(b n) f d -> b n f d', b=x.shape[0])  # [b, n, f, d]
        return out

class DualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.temporal_attn = TemporalAttention(d_model, n_heads, dropout)
        self.freq_attn = FreqAttention(d_model, n_heads, dropout)
        self.fusion = LearnableFusion(2, d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.temporal_norm = RMSNorm(d_model)
        self.freq_norm = RMSNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [b, n, f, d]
        temp_out = self.temporal_attn(self.temporal_norm(x))  # [b, n, f, d]
        freq_out = self.freq_attn(self.freq_norm(x))          # [b, n, f, d]

        fused = self.fusion(temp_out, freq_out)               # [b, n, f, d]
        x = x + self.dropout(fused)

        ffn_out = self.ffn(self.ffn_norm(x))                  # [b, n, f, d]
        x = x + self.dropout(ffn_out)
        return x

class StackDualAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            DualAttentionBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: [b, n, f, d]
        for layer in self.layers:
            x = layer(x)
        return x

