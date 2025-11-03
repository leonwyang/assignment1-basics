from __future__ import annotations
from pathlib import Path
from collections.abc import Iterator
import json

import torch
import numpy as np
from jaxtyping import Int
from torch import Tensor

try:
    from cs336_basics.logger import logger
except Exception:
    import logging as logger  # fallback

class MemoryMappedDataset:
    def __init__(
        self,
        path_or_ds: str | Path | np.ndarray,
        context_length: int,
        device: str = "cpu",
        seed: int | None = None,
    ):
        """
        Supports:
          - Raw int*.bin with a sibling .meta.json (from encode_corpus_parallel.py)
          - .npy files (np.save) with mmap_mode='r'
          - An in-memory ndarray
        """
        self.context_length = int(context_length)
        self.device = device
        self.seed = seed

        if isinstance(path_or_ds, np.ndarray):
            self.ds = path_or_ds
            self.total_length = int(self.ds.shape[0])
            logger.info(f"Dataset (ndarray) length: {self.total_length}")
            return

        path = Path(path_or_ds)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

        if path.suffix == ".npy":
            # Standard numpy file (memmap)
            self.ds = np.load(path, mmap_mode="r", allow_pickle=True)  # type: ignore
            self.total_length = int(self.ds.shape[0])
            logger.info(f"Dataset (.npy memmap) length: {self.total_length}")
        elif path.suffix == ".bin":
            # Raw binary from our encoder: need dtype + length from .meta.json
            meta_path = Path(str(path) + ".meta.json")
            if not meta_path.exists():
                # also allow "train.bin.meta.json" naming
                alt = path.with_suffix(path.suffix + ".meta.json")
                if alt.exists():
                    meta_path = alt
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"Missing metadata for raw .bin: {meta_path} (expected alongside {path})"
                )
            meta = json.loads(meta_path.read_text())
            length = int(meta["length"])
            dtype_str = str(meta["dtype"])
            dtype = getattr(np, dtype_str)
            self.ds = np.memmap(path, mode="r", dtype=dtype, shape=(length,))
            self.total_length = length
            logger.info(f"Dataset (.bin memmap) length: {self.total_length}, dtype: {dtype_str}")
        else:
            # Try loading as .npy anyway; otherwise advise
            try:
                self.ds = np.load(path, mmap_mode="r")  # type: ignore
                self.total_length = int(self.ds.shape[0])
                logger.info(f"Dataset (np.load) length: {self.total_length}")
            except Exception as e:
                raise ValueError(
                    f"Unsupported dataset extension '{path.suffix}'. "
                    f"Use a raw '.bin' with a '.meta.json' (from the encoder) or a '.npy'."
                ) from e

        if self.total_length <= self.context_length:
            raise ValueError(
                f"Dataset too short: total_length={self.total_length}, "
                f"context_length={self.context_length}"
            )

    def __len__(self) -> int:
        # last index is i + context_length + 1, so starts go up to total_length - (context_length + 1)
        return self.total_length - (self.context_length + 1)

    def __getitem__(self, i: int) -> tuple[Int[Tensor, "context"], Int[Tensor, "context"]]:
        """Return one training example (input/target) starting at index i."""
        i_end = i + self.context_length + 1
        # Convert to int64 torch tensors (typical token dtype)
        # Keep zero-copy from memmap where possible (astype(copy=False)).
        chunk = self.ds[i:i_end].astype(np.int64, copy=False)
        inputs = torch.from_numpy(chunk[:-1]).to(device=self.device)
        targets = torch.from_numpy(chunk[1:]).to(device=self.device)
        return inputs, targets

    def get_batch_iterator(
        self, batch_size: int, *, shuffle: bool = True
    ) -> Iterator[tuple[Int[Tensor, "bs context"], Int[Tensor, "bs context"]]]:
        """
        Yields batches of shape [bs, context_length] on self.device.
        """
        assert batch_size > 0
        n = len(self)
        starts = np.arange(0, n, dtype=np.int64)
        if shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(starts)

        # Stack slices into numpy arrays; each slice preserves memmap zero-copy to the extent possible.
        for i in range(0, n, batch_size):
            s = starts[i : i + batch_size]
            x_np = np.stack(
                [self.ds[j : j + self.context_length].astype(np.int64, copy=False) for j in s]
            )
            y_np = np.stack(
                [self.ds[j + 1 : j + 1 + self.context_length].astype(np.int64, copy=False) for j in s]
            )
            x = torch.from_numpy(x_np).to(self.device)
            y = torch.from_numpy(y_np).to(self.device)
            yield x, y
