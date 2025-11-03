from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum

def softmax(x: Tensor, dim: int = 0, temperature: float = 1.0) -> Tensor:
    o: Tensor = x - x.max(dim=dim, keepdim=True)[0]
    assert temperature > 0, "temperature must be more than 0"
    if temperature != 1.0:
        o /= temperature
    return o.exp() / o.exp().sum(dim=dim, keepdim=True)

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.w = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_type = x.dtype
        x = x.to(torch.float32)
        rms_inv = torch.rsqrt((x*x).mean(dim=-1, keepdim=True) + self.eps)
        res = rms_inv * x * self.w
        return res.to(x_type)


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        sigma = (2.0 / (in_features + out_features)) ** 0.5
        sigma_t = torch.tensor(sigma, device=device, dtype=dtype)

        # Weight: (out_features, in_features)
        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w, mean=0.0, std=sigma, a=-3, b=3)
        # w.clamp_(-3 * sigma_t, 3 * sigma_t)
        self.weight = torch.nn.Parameter(w)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... in_dim, out_dim in_dim -> ... out_dim')


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        print(type(num_embeddings))
        print(embedding_dim)
        w = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w, mean=0, std=1, a=-3, b=3)
        self.weight = torch.nn.Parameter(w)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()
        return self.weight[token_ids]
    
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device, dtype):
        super().__init__()
        self.linear1 = Linear(in_features = d_model, out_features=d_ff, device=device, dtype=dtype)
        self.linear3 = Linear(in_features = d_model, out_features=d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(in_features = d_ff, out_features=d_model, device=device, dtype=dtype)

    def forward(self, x):
        w1_x = self.linear1(x)
        w3_x = self.linear3(x)
        silu = w1_x * torch.sigmoid(w1_x)
        return self.linear2(silu * w3_x)
    



class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE (Rotary Positional Embedding)

    Args:
        theta: Θ base for the inverse frequency geometric sequence (e.g., 10_000.0).
        d_k: dimensionality of the last axis of x (must be even).
        max_seq_len: maximum sequence length you will pass in `token_positions`.
        device: optional device to initialize buffers on.

    Forward:
        x: tensor of shape (..., seq_len, d_k)
        token_positions: Long tensor of shape (..., seq_len) with absolute positions for each token.

    Returns:
        Tensor with the same shape as x, with RoPE applied to the last dimension.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE (got {d_k}).")

        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)

        d_half = d_k // 2

        # Inverse frequencies: theta^{-2i/d_k} for i = 0..d_half-1
        # (equivalently: exp(-log(theta) * 2i/d_k))
        inv_freq = self.theta ** (-2 * torch.arange(d_half, device=device).float() / d_k)
        # Precompute cos/sin for all positions up to max_seq_len
        positions = torch.arange(max_seq_len, device=device).float()  # (L,)
        # (L, d_half)
        freq = torch.outer(positions, inv_freq)
        cos = torch.cos(freq)
        sin = torch.sin(freq)

        # Register as buffers so they move with .to() and are saved in state_dict
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to x using token_positions to slice precomputed cos/sin.

        Shapes:
            x: (..., seq_len, d_k)
            token_positions: (..., seq_len) with values in [0, max_seq_len)
        """
        if token_positions.dtype != torch.long:
            token_positions = token_positions.long()

        if torch.any(token_positions < 0) or torch.any(token_positions >= self.max_seq_len):
            raise IndexError(
                "token_positions must be in the range [0, max_seq_len). "
                f"Got min={int(token_positions.min())}, max={int(token_positions.max())}, "
                f"max_seq_len={self.max_seq_len}."
            )

        # Ensure buffers are on the same device/dtype as x
        cos = self.cos.to(device=x.device, dtype=x.dtype)
        sin = self.sin.to(device=x.device, dtype=x.dtype)

        # Gather cos/sin for the provided positions.
        # self.cos/self.sin: (L, d_half)
        # token_positions: (..., seq_len)
        # cos_pos/sin_pos: (..., seq_len, d_half)
        cos_pos = cos[token_positions]  # advanced indexing
        sin_pos = sin[token_positions]

        # Split last dim of x into pairs: [x_even, x_odd]
        # x: (..., seq_len, d_k) -> (..., seq_len, d_half, 2)
        d_half = self.d_k // 2
        x_view = x.view(*x.shape[:-1], d_half, 2)
        x_even = x_view[..., 0]  # (..., seq_len, d_half)
        x_odd = x_view[..., 1]   # (..., seq_len, d_half)

        # Apply rotation:
        # [x_even; x_odd] -> [x_even * cos - x_odd * sin ; x_odd * cos + x_even * sin]
        rot_even = x_even * cos_pos - x_odd * sin_pos
        rot_odd = x_odd * cos_pos + x_even * sin_pos

        # Re-interleave into (..., seq_len, d_k)
        out = torch.stack((rot_even, rot_odd), dim=-1).reshape(*x.shape)
        return out

