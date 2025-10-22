import math

import torch
import torch.nn as nn
import torch.nn.functional


class AttnPool1d(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, num_queries: int = 2, attn_dropout: float = 0.0):
        super().__init__()
        self.num_queries = num_queries
        self.key_proj = nn.Linear(d_in, d_hidden, bias=True)
        self.query = nn.Parameter(torch.randn(num_queries, d_hidden) / math.sqrt(d_hidden))
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D_in]
        B, N, _ = x.shape
        k = self.key_proj(x)                        # [B, N, Dh]
        # scores: [B, K, N]
        scores = torch.einsum('bnd,kd->bkn', k, self.query) / math.sqrt(k.shape[-1])
        attn = torch.nn.functional.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        # pooled: [B, K, Dh]
        pooled = torch.einsum('bkn,bnd->bkd', attn, k)
        return pooled

class MWPD(nn.Module):
    def __init__(
        self,
        patch_num: int,
        d_model: int,
        pred_lens: list,
        d_ff: int,
        dropout: float = 0.1,
        bottleneck_ratio: float = 0.5,
    ):
        super().__init__()
        self.patch_num = patch_num
        self.d_model = d_model
        self.pred_lens = pred_lens
        self.num_bands = len(pred_lens)
        self.dropout = nn.Dropout(dropout)

        dh = int(round(d_model * bottleneck_ratio))
        self.num_queries = int(1/bottleneck_ratio)
        self.in_norm = nn.LayerNorm(d_model)
        self.pool = AttnPool1d(d_in=d_model, d_hidden=dh, num_queries=self.num_queries, attn_dropout=dropout)
        trunk_dim = self.num_queries * dh
        self.trunk_dim = trunk_dim
        self.out_norm = nn.LayerNorm(trunk_dim)
        self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(trunk_dim),
                    nn.Linear(trunk_dim, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, L_out),
                )
                for L_out in pred_lens
            ])

    def forward(self, x_for_all_bands: torch.Tensor):
        """
        x_for_all_bands: [B, N, F, D]
        Returns: list of [B, 1, pred_len_i] for each band
        """
        B, N, F, D = x_for_all_bands.shape
        assert F == self.num_bands and D == self.d_model

        predictions = []
        for i in range(F):
            x = x_for_all_bands[:, :, i, :]         # [B, N, D]
            x = self.in_norm(x)
            pooled = self.pool(x)                    # [B, K, Dh]
            trunk = pooled.reshape(B, -1)            # [B, K*Dh]
            trunk = self.out_norm(trunk)
            trunk = self.dropout(trunk)

            pred = self.heads[i](trunk)              # [B, L_out]
            predictions.append(pred.unsqueeze(1))    # [B, 1, L_out]

        return predictions
