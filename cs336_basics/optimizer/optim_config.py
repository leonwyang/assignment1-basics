from __future__ import annotations
from dataclasses import dataclass
from cs336_basics.optimizer import adamw
from typing import Protocol, Literal, Union, Iterable, Mapping, Any
import torch

class OptimFactory(Protocol):
    type:str
    def make(self, params)->torch.optim.Optimizer:...

@dataclass(frozen=True)
class AdamWFactory(OptimFactory):
    type: str="adamw"
    lr: float = 7e-3
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 1e-7
    eps: float = 1e-8

    def make(self, params):
        return adamw.AdamW(params, self.lr, self.betas, self.eps, self.weight_decay)


AnyOptimConfig = Union[AdamWFactory]
