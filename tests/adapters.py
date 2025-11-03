from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import IO, Any, BinaryIO
import re
from collections import Counter
from typing import Any, Iterable, Iterator, List, Tuple, Dict, Optional
from cs336_basics.tokenizer import train_bpe


class BPETokenizer:
    def __init__(
        self,
        id_to_bytes: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        # Core maps
        self.id_to_bytes: Dict[int, bytes] = dict(id_to_bytes)
        self.bytes_to_id: Dict[bytes, int] = {b: i for i, b in self.id_to_bytes.items()}

        # Merge ranks: lower index = higher priority
        self.ranks: Dict[Tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

        # Special tokens
        self.special_tokens: List[str] = special_tokens or []
        if self.special_tokens:
            escaped = [re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)]
            self._special_re = re.compile("(" + "|".join(escaped) + ")")
        else:
            self._special_re = None

        # Require byte-level base (gpt2/tiktoken style)
        missing = [bytes([b]) for b in range(256) if bytes([b]) not in self.bytes_to_id]
        if missing:
            raise ValueError(
                f"Vocabulary is missing {len(missing)} single-byte entries (byte-level base required)."
            )

        # Ensure specials exist as UTF-8 bytes in vocab
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in self.bytes_to_id:
                raise ValueError(f"Special token {tok!r} not found in vocab.")

        # Cache the single '\n' id (used for newline isolation)
        self._nl_id = self.bytes_to_id[b"\n"]

    # ---------- Public API ----------
    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        return list(self._encode_iter(text))

    def encode_iterable(self, source: Iterable[str]) -> Iterator[int]:
        """Yield token IDs lazily from an iterable of strings (e.g., an open file)."""
        for chunk in source:
            yield from self._encode_iter(chunk)

    def decode(self, ids: Iterable[int]) -> str:
        b = b"".join(self.id_to_bytes[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    # ---------- Internals ----------
    def _encode_iter(self, text: str) -> Iterator[int]:
        """Handle specials and newline isolation; run BPE on non-newline spans."""
        if not text:
            return
        if self._special_re is None:
            # No specials: encode entire text
            yield from self._encode_without_specials(text)
            return

        # Split into [segment, special, segment, ...]
        parts = self._special_re.split(text)
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                # Emit the special token id
                yield self.bytes_to_id[part.encode("utf-8")]
            else:
                yield from self._encode_without_specials(part)

    def _encode_without_specials(self, segment: str) -> Iterator[int]:
        if not segment:
            return
        data = segment.encode("utf-8")
        i = 0
        n = len(data)
        while i < n:
            try:
                j = data.index(b"\n", i)
            except ValueError:
                # no more newlines: BPE the tail
                if i < n:
                    yield from self._bpe_bytes(data[i:n])
                break

            # bytes before the newline
            if j > i:
                yield from self._bpe_bytes(data[i:j])

            # find the end of this run of consecutive newlines
            k = j
            while k < n and data[k] == 0x0A:  # b"\n"
                k += 1

            if k == n:
                # TRAILING newline run: allow BPE merges (e.g., "\n\n" -> 628)
                yield from self._bpe_bytes(data[j:k])
                break
            else:
                # MIDDLE newline run: emit one '\n' token per newline (no merging)
                count = k - j
                for _ in range(count):
                    yield self._nl_id
                i = k

    def _bpe_bytes(self, data: bytes) -> Iterator[int]:
        """Standard BPE on the provided bytes (no crossing newline boundaries)."""
        if not data:
            return
        tokens: List[bytes] = [bytes([b]) for b in data]

        while True:
            best_rank = None
            best_pair = None
            i = 0
            # Find best-ranked adjacent pair
            while i < len(tokens) - 1:
                pair = (tokens[i], tokens[i + 1])
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
                i += 1

            if best_pair is None:
                break

            # Merge all occurrences of best_pair
            new_tokens: List[bytes] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        for tok in tokens:
            try:
                yield self.bytes_to_id[tok]
            except KeyError as e:
                missing = e.args[0]
                raise KeyError(
                    f"Final token bytes {missing!r} not found in vocab. "
                    "Ensure your vocab includes tokens for all BPE merges."
                ) from None

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch import nn
from einops import einsum, rearrange


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




def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    layer = Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
    layer.load_state_dict({'weight': weights}, strict=True)
    # with torch.inference_mode():
    out = layer(in_features)
    return out


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        w = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w, mean=0, std=1, a=-3, b=3)
        self.weight = torch.nn.Parameter(w)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()
        return self.weight[token_ids]
    
def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    emb = Embedding(vocab_size, d_model, weights.device, weights.dtype)
    state_dict = {'weight': weights}
    emb.load_state_dict(state_dict)
    return emb(token_ids)


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



def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    swiglu_layer = SwiGLU(d_model, d_ff)
    with torch.no_grad():
        swiglu_layer.linear1.load_state_dict({'weight': w1_weight})
        swiglu_layer.linear2.load_state_dict({'weight': w2_weight})
        swiglu_layer.linear3.load_state_dict({'weight': w3_weight})
        return swiglu_layer(in_features)

import math
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
    attn = run_softmax(QK_scaled, dim=-1) # '... n m'
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




def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mha = MultiHeadedSelfAttention(d_model, num_heads, in_features.device, in_features.dtype, apply_rope=False)
    state_dict = {
        'Q.weight': q_proj_weight,
        'K.weight': k_proj_weight,
        'V.weight': v_proj_weight,
        'output_proj.weight': o_proj_weight
    }
    mha.load_state_dict(state_dict, strict=True)
    o = mha(in_features)
    return o


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mha = MultiHeadedSelfAttention(d_model, num_heads, in_features.device, in_features.dtype, max_seq_len=max_seq_len, theta=theta, apply_rope=True)
    state_dict = {
        'Q.weight': q_proj_weight,
        'K.weight': k_proj_weight,
        'V.weight': v_proj_weight,
        'output_proj.weight': o_proj_weight
    }
    mha.load_state_dict(state_dict, strict=True)
    o = mha(in_features, token_positions)
    return o


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

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope_emb = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_query_or_key.device)
    return rope_emb(in_query_or_key, token_positions)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, device, dtype, apply_rope=True, max_seq_len=2048, theta=10000):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.mha = MultiHeadedSelfAttention(d_model, num_heads, device, dtype, apply_rope=apply_rope, max_seq_len=max_seq_len, theta=theta)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.rms_norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rms_norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x):
        x = self.mha(self.rms_norm1(x)) + x
        x = self.ffn(self.rms_norm2(x)) + x
        return x

        
        


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    transformer_block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                                          device=in_features.device,
                                          dtype=in_features.dtype,
                                          apply_rope=True,
                                          max_seq_len=max_seq_len,
                                          theta=theta)
    state_dict = {
        'mha.Q.weight': weights['attn.q_proj.weight'],
        'mha.K.weight': weights['attn.k_proj.weight'],
        'mha.V.weight': weights['attn.v_proj.weight'],
        'mha.output_proj.weight': weights['attn.output_proj.weight'],
        'rms_norm1.w':  weights['ln1.weight'],
        'rms_norm2.w':  weights['ln2.weight'],
        'ffn.linear1.weight': weights['ffn.w1.weight'],
        'ffn.linear2.weight': weights['ffn.w2.weight'],
        'ffn.linear3.weight': weights['ffn.w3.weight'],
    }
    transformer_block.load_state_dict(state_dict)
    return transformer_block(in_features)
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device, dtype):
        super().__init__()
        self.context_length = context_length
        self.emb = Embedding(vocab_size, embedding_dim=d_model)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.transformer_layers = []
        self.output_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.transformer_layers = nn.ModuleList([TransformerBlock(d_model=d_model,
                                                        num_heads=num_heads,
                                                        d_ff=d_ff,
                                                        device=device,
                                                        dtype=dtype,
                                                        apply_rope=True,
                                                        max_seq_len=context_length,
                                                        theta=rope_theta) for _ in range(num_layers)])
    
        

    def forward(self, token_ids):
        x = self.emb(token_ids)
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.output_norm(x)
        x = self.output_linear(x)
        return x



def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    transformer_lm = TransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=d_model,
                       num_layers=num_layers, num_heads=num_heads,
                       d_ff=d_ff, rope_theta=rope_theta, device=in_indices.device, dtype=torch.float32)
    state_dict = {'emb.weight': weights['token_embeddings.weight']}

    for i in range(num_layers):
        state_dict[f'transformer_layers.{i}.mha.Q.weight'] = weights[f'layers.{i}.attn.q_proj.weight']
        state_dict[f'transformer_layers.{i}.mha.K.weight'] = weights[f'layers.{i}.attn.k_proj.weight']
        state_dict[f'transformer_layers.{i}.mha.V.weight'] = weights[f'layers.{i}.attn.v_proj.weight']
        state_dict[f'transformer_layers.{i}.mha.output_proj.weight'] = weights[f'layers.{i}.attn.output_proj.weight']
        state_dict[f'transformer_layers.{i}.rms_norm1.w'] = weights[f'layers.{i}.ln1.weight']
        state_dict[f'transformer_layers.{i}.rms_norm2.w'] = weights[f'layers.{i}.ln2.weight']
        state_dict[f'transformer_layers.{i}.ffn.linear1.weight'] = weights[f'layers.{i}.ffn.w1.weight']
        state_dict[f'transformer_layers.{i}.ffn.linear2.weight'] = weights[f'layers.{i}.ffn.w2.weight']
        state_dict[f'transformer_layers.{i}.ffn.linear3.weight'] = weights[f'layers.{i}.ffn.w3.weight']
    state_dict[f'output_norm.w'] = weights[f'ln_final.weight']
    state_dict[f'output_linear.weight'] = weights['lm_head.weight']
    transformer_lm.load_state_dict(state_dict, strict=True)
    return transformer_lm(in_indices)


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




def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rms_layer = RMSNorm(d_model, eps)
    rms_layer.load_state_dict({'w': weights})
    return rms_layer(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError

import numpy as np

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    assert context_length < dataset.shape[0]
    max_start_idx = dataset.shape[0] - context_length
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    input_seqs = []
    output_seqs = []
    for idx in start_indices:
        input_seq = dataset[idx:idx+context_length]
        output_seq = dataset[idx+1:idx+context_length+1]
        input_seqs.append(input_seq)
        output_seqs.append(output_seq)
    return torch.from_numpy(np.array(input_seqs)).long().to(device=device), torch.from_numpy(np.array(output_seqs)).long().to(device=device)



def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max_val, _ = in_features.max(dim=dim, keepdim=True)
    in_features_sub_max = in_features - max_val
    return torch.exp(in_features_sub_max) / torch.sum(torch.exp(in_features_sub_max), dim=dim, keepdim=True)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    lse = torch.logsumexp(inputs, dim=-1)  # (batch,)
    target_logits = inputs[torch.arange(inputs.shape[0]), targets]  # (batch,)
    return torch.mean(lse - target_logits)

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    eps = 1e-6
    grads = []
    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad)
    total_norm = 0
    for grad in grads:
        total_norm += grad.norm().item()**2
    total_norm = total_norm**0.5
    if total_norm > max_l2_norm:
        coeff = max_l2_norm / total_norm
        for grad in grads:
            grad.mul_(coeff)



from typing import Callable

import math
from typing import Optional, Callable
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-2):
        if lr <= 0: 
            raise ValueError("lr must be > 0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # step number (increment first, then use)
                state["t"] += 1
                t = state["t"]

                # Bias-corrected step size
                lr_adj = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                # Update moments (in-place)
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # Parameter update (in-place)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-lr_adj)
                if wd != 0:
                    p.mul_(1 - lr * wd)

        return loss
    

# class AdamW(torch.optim.Optimizer):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay = 1e-2):
#         assert lr > 0
#         defaults = {'lr': lr}
#         super().__init__(params, defaults)
#         self.eps = eps
#         self.beta1 = betas[0]
#         self.beta2 = betas[1]
#         self.weight_decay = weight_decay

#     @torch.no_grad()
#     def step(self, closure: Optional[Callable] = None):
#         loss = None if closure is None else closure()
#         for group in self.param_groups:
#             lr = group["lr"] # Get the learning rate.
#             for p in group["params"]:
#                 state = self.state[p]
#                 t = state.get("t", 1) 
#                 if len(state) == 0:
#                     state["t"] = 0
#                     state["exp_avg"] = torch.zeros_like(p)
#                     state["exp_avg_sq"] = torch.zeros_like(p)

#                 exp_avg = state["exp_avg"]
#                 exp_avg_sq = state["exp_avg_sq"]

#                 lr_adj = lr * math.sqrt(1-(self.beta2)**t) / (1-self.beta1**t)
#                 state["t"] = t + 1
#                 if p.grad is None:
#                     continue
#                 g = p.grad.data
#                 # Update moments (in-place)
#                 exp_avg.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
#                 exp_avg_sq.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)

#                 # Parameter update (in-place)
#                 denom = exp_avg_sq.sqrt().add_(self.eps)
#                 p.addcdiv_(exp_avg, denom, value=-lr_adj)

#         return loss


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 1/2 * (1 + math.cos((it - warmup_iters)/(cosine_cycle_iters - warmup_iters) * math.pi ))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint_dict, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint_state_dict = torch.load(src)
    model.load_state_dict(checkpoint_state_dict['model_state_dict'])
    optimizer.load_state_dict(checkpoint_state_dict['optimizer_state_dict'])
    it = checkpoint_state_dict['iteration']
    return it


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    from cs336_basics.tokenizer_back2 import Tokenizer
    return Tokenizer(vocab, merges, special_tokens)
    # return BPETokenizer(vocab, merges, special_tokens)



def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # from cs336_basics.bpe import BPE
    # bpe = BPE(special_tokens=special_tokens, vocab_size=vocab_size)
    # vocab, merges = bpe.train(input_path, **kwargs)
    # return vocab, merges
    from cs336_basics.tokenizer import train_bpe
    return train_bpe(input_path, vocab_size, special_tokens)


def _assemble_vocab(
    learned: Dict[int, bytes],
    specials_bytes: List[bytes],
    vocab_size: int,
) -> Dict[int, bytes]:
    # First 256: single bytes
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # Learned tokens in creation order (ids are increasing)
    for tid in sorted((k for k in learned.keys() if k >= 256)):
        if len(vocab) >= vocab_size:
            break
        vocab[tid] = learned[tid]

    # Append special tokens if there’s room; avoid duplicates
    next_id = max(vocab.keys(), default=-1) + 1
    existing_vals = set(vocab.values())
    for sb in specials_bytes:
        if len(vocab) >= vocab_size:
            break
        if sb in existing_vals:
            continue
        vocab[next_id] = sb
        existing_vals.add(sb)
        next_id += 1

    # Trim if somehow over
    if len(vocab) > vocab_size:
        keep = sorted(vocab.keys())[:vocab_size]
        vocab = {i: vocab[i] for i in keep}

    return vocab
