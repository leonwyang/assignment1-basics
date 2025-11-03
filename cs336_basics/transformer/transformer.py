from cs336_basics.transformer.attention import MultiHeadedSelfAttention
from cs336_basics.transformer.layers import SwiGLU, RMSNorm, Embedding, Linear
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum

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
    

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device, dtype):
        super().__init__()
        self.context_length = context_length
        self.emb = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
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