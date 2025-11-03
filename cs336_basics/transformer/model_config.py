from dataclasses import dataclass
from cs336_basics.transformer import transformer, layers
from typing import Protocol, Literal, Union, Iterable, Mapping, Any
import torch
class TransformerConfigProtocol(Protocol):
    def make(self, device, dtype) -> transformer.TransformerLM: ...

# @dataclass
# class TransformerConfig:
#     vocab_size: int=60000
#     d_model: int=1024
#     num_heads: int=128
#     num_layers: int=30
#     d_ff: int=2048
#     apply_rope: bool=True
#     context_length: int=2048
#     rope_theta: int=10000
#     def make(self, device, dtype):
#         print("inside", self.vocab_size)
#         return transformer.TransformerLM(
#             vocab_size=self.vocab_size,
#             context_length=self.context_length,
#             d_model=self.d_model,
#             num_layers=self.num_layers,
#             num_heads=self.num_heads,
#             d_ff=self.d_ff,
#             rope_theta=self.rope_theta,
#             device=device,
#             dtype=dtype
#         )

@dataclass
class TransformerConfig:
    vocab_size: int=10000
    d_model: int=512
    num_heads: int=16
    num_layers: int=4
    d_ff: int=1344
    apply_rope: bool=True
    context_length: int=256
    rope_theta: int=10000
    def make(self, device, dtype):
        print("inside", self.vocab_size)
        return transformer.TransformerLM(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            device=device,
            dtype=dtype
        )