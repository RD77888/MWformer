import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, d_model, max_len=100, base=10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        assert d_model % 2 == 0, "d_model must be even for RoPE"

        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        t = torch.arange(self.max_len, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)  # [max_len, d_model//2]

        cos_cached = torch.cos(freqs)
        sin_cached = torch.sin(freqs)
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)

    def forward(self, x, seq_offset=0):
        batch_size, seq_len, num_heads, head_dim = x.shape
        required_len = seq_offset + seq_len
        if required_len > self.max_len:
            raise ValueError(
                f"Sequence length {required_len} exceeds the pre-computed max_len {self.max_len}. "
                "You must re-initialize the RoPE module with a larger `max_len`."
            )

        cos = self.cos_cached[seq_offset:required_len].to(x.dtype)
        sin = self.sin_cached[seq_offset:required_len].to(x.dtype)

        return self._apply_rope(x, cos, sin)

    def _apply_rope(self, x, cos, sin):
        x_even = x[..., ::2]  # [batch, seq_len, num_heads, d_model//2]
        x_odd = x[..., 1::2]  # [batch, seq_len, num_heads, d_model//2]

        # cos/sin [seq_len, d_model//2] -> [1, seq_len, 1, d_model//2] for broadcasting
        x_rotated_even = x_even * cos[None, :, None, :] - x_odd * sin[None, :, None, :]
        x_rotated_odd = x_even * sin[None, :, None, :] + x_odd * cos[None, :, None, :]

        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated


