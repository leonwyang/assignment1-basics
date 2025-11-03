from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from cs336_basics.dataloader.mmdata import MemoryMappedDataset

@dataclass
class MMDatasetConfig:
    train_path: str | Path = "cs336_basics/tokenized_parallel/train.bin"
    validation_path: str | Path = "cs336_basics/tokenized_parallel/val.bin"
    batch_size: int = 8
    val_batch_size: int = 8
    context_length: int = 256
    seed: int = 42
    device: str = "mps"

    def make(self):
        train = MemoryMappedDataset(
            path_or_ds=self.train_path,
            context_length=self.context_length,
            device=self.device,
            seed=self.seed,
        )
        val = MemoryMappedDataset(
            path_or_ds=self.validation_path,
            context_length=self.context_length,
            device=self.device,
            seed=self.seed,
        )
        return train, val