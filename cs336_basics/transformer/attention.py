import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int, Bool
from tqdm.auto import tqdm
from einops import einsum
import math

from cs336_basics.transformer.layers import Linear, softmax, RotaryPositionalEmbedding

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    QK = einsum(Q, K, '... n d_k, ... m d_k -> ... n m')
    QK_scaled = QK * (1.0 / math.sqrt(d_k))
    if mask is not None:
        QK_scaled = QK_scaled.masked_fill(~mask, -torch.inf)
    attn = softmax(QK_scaled, dim=-1) # '... n m'
    return einsum(attn, V, '... n m, ... m d_v -> ... n d_v')

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, device, dtype, apply_rope=False, theta=10000, max_seq_len=4096):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.sub_d_model = d_model // num_heads
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.apply_rope = apply_rope
        if apply_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.sub_d_model, max_seq_len=max_seq_len)
        self.Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, positions=None):
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        # assume casual
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).to(torch.bool)
        # dim: batch, num_heads, T, sub_d_model
        q = q.view(*batch_shape, seq_len, self.num_heads, self.sub_d_model).transpose(-3, -2)
        k = k.view(*batch_shape, seq_len, self.num_heads, self.sub_d_model).transpose(-3, -2)
        v = v.view(*batch_shape, seq_len, self.num_heads, self.sub_d_model).transpose(-3, -2)
        if self.apply_rope:
            if positions is None:
                positions = torch.arange(seq_len)

            # Broadcast positions to match shape (..., num_heads, seq_len)
            pos = positions.unsqueeze(-2).expand(*batch_shape, self.num_heads, seq_len)
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        o = run_scaled_dot_product_attention(q, k, v, causal_mask)
        o = o.transpose(-3, -2).contiguous().view(*batch_shape, seq_len, self.d_model)
        o = self.output_proj(o)
        return o