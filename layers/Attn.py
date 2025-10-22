import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RMSnorm import RMSNorm
from layers.ROPE import RoPE


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim: int ,d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.head_dim = d_model // n_heads

        # Query, Key, Value projections
        self.wq = nn.Linear(in_dim, d_model)
        self.wk = nn.Linear(in_dim, d_model)
        self.wv = nn.Linear(in_dim, d_model)
        self.wo = nn.Linear(d_model, in_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        self.rope = RoPE(d_model // n_heads)

        # RMSNorm for Q and K
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, q, k, v, causal=False, mask=None, eps=1e-6):
        B, Nq, _ = q.shape
        _, Nk, _ = k.shape

        q_proj = self.wq(q)  # [B, Nq, d_model]
        k_proj = self.wk(k)  # [B, Nk, d_model]
        v_proj = self.wv(v)  # [B, Nk, d_model]

        q = q_proj.view(B, Nq, self.n_heads, self.head_dim)  # [B, Nq, H, Dh]
        k = k_proj.view(B, Nk, self.n_heads, self.head_dim)  # [B, Nk, H, Dh]

        q = self.rope(q)  # [B, Nq, H, Dh]
        k = self.rope(k)  # [B, Nk, H, Dh]

        q = q.transpose(1, 2)  # [B, H, Nq, Dh]
        k = k.transpose(1, 2)  # [B, H, Nk, Dh]
        v = v_proj.view(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, Nk, Dh]

        # 5. RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_mask = None
        if causal:
            causal_mask = torch.triu(
                torch.ones(Nq, Nk, device=q.device, dtype=torch.bool),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            attn_mask = causal_mask

        if mask is not None:
            if attn_mask is None:
                attn_mask = mask.unsqueeze(0).unsqueeze(0)
            else:
                attn_mask = attn_mask | mask.unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            scale=self.scale
        )  # [B, H, Nq, Dh]

        out = out.transpose(1, 2).contiguous()  # [B, Nq, H, Dh]
        out = out.view(B, Nq, self.d_model)  # [B, Nq, d_model]
        out = self.wo(out)  # [B, Nq, in_dim]
        out = self.dropout(out)

        return out, None
