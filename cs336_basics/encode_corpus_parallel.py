#!/usr/bin/env python3
"""
Parallel TinyStories encoder with progress tracking.

Outputs (under --out-dir):
  train.bin, train.bin.meta.json
  val.bin,   val.bin.meta.json

Memmap later with:
  meta = json.load(open("train.bin.meta.json"))
  arr  = np.memmap("train.bin", mode="r", dtype=np.int32, shape=(meta["length"],))
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---- Make prints show up immediately (even under uv/multiprocessing) ----
try:
    sys.stdout.reconfigure(line_buffering=True)  # py3.7+
except Exception:
    pass

# ---- Import your boundary finder ----
sys.path.append(".")
from cs336_basics.pretokenization_example import find_chunk_boundaries  # noqa: E402

# ---- Import your Tokenizer (must expose .from_files) ----
try:
    from tinystories_tokenizer import Tokenizer  # preferred module name
except Exception:
    try:
        from tokenizer import Tokenizer  # fallback name
    except Exception as e:
        raise ImportError(
            "Could not import Tokenizer. Put your Tokenizer class in "
            "`tinystories_tokenizer.py` or `tokenizer.py` next to this script. "
            "It must define Tokenizer.from_files(vocab_pkl, merges_pkl, special_tokens)."
        ) from e


# =========================
# Multiprocessing machinery
# =========================

_TOK = None  # tokenizer instance loaded once per worker


def _worker_init(vocab_pkl: str, merges_pkl: str, special_tokens: List[str]):
    """Initializer run once per worker; loads tokenizer into a global."""
    global _TOK
    _TOK = Tokenizer.from_files(vocab_pkl, merges_pkl, special_tokens=special_tokens)


def _encode_chunk_worker(args) -> Tuple[int, str, int]:
    """
    Worker: encode a [start, end) byte range of a file.
    Writes a temp .bin of ids (dtype chosen by caller).
    Returns (chunk_index, tmp_path, num_tokens).
    """
    idx, in_path, start, end, np_dtype_str = args
    dtype = getattr(np, np_dtype_str)

    # Read chunk bytes, then decode once
    with open(in_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    text = data.decode("utf-8", errors="ignore")

    # Stream-encode line by line to limit peak memory
    ids: List[int] = []
    for line in text.splitlines(keepends=True):
        ids.extend(_TOK.encode(line))

    # Dump to a temp file
    fd, tmp_path = tempfile.mkstemp(prefix=f"enc_{idx:06d}_", suffix=".bin")
    os.close(fd)
    np.asarray(ids, dtype=dtype).tofile(tmp_path)
    return idx, tmp_path, len(ids)


# =========================
# Utilities
# =========================

def _write_meta_json(bin_path: Path, length: int, dtype: str):
    meta = {"length": int(length), "dtype": dtype}
    meta_path = Path(str(bin_path) + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _concatenate_temp_bins(pieces: List[Tuple[int, str, int]], out_path: Path, np_dtype_str: str) -> int:
    """Concatenate temp .bin pieces in order into `out_path` using a memmap destination."""
    dtype = getattr(np, np_dtype_str)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pieces_sorted = sorted(pieces, key=lambda x: x[0])
    total = sum(n for _, _, n in pieces_sorted)
    itemsize = np.dtype(dtype).itemsize

    # Pre-size output
    with open(out_path, "wb") as f:
        f.truncate(total * itemsize)

    out_mm = np.memmap(out_path, mode="r+", dtype=dtype, shape=(total,))
    offset = 0

    for idx, tmp_path, n in pieces_sorted:
        tmp_mm = np.memmap(tmp_path, mode="r", dtype=dtype, shape=(n,))
        out_mm[offset: offset + n] = tmp_mm[:]
        offset += n
        del tmp_mm
        os.remove(tmp_path)

    out_mm.flush()
    del out_mm
    return total


def _find_boundaries_or_warn(in_txt_path: Path, workers: int, boundary_token: str) -> List[int]:
    """Find byte-range boundaries; warn if boundary token is missing or unquoted."""
    bt = boundary_token
    if bt and ("<" in bt or ">" in bt) and not (bt.startswith("<") and bt.endswith(">")):
        # Mild heuristic for weirdly passed token; user should quote in shell.
        print(f"[warning] boundary token looks odd: {bt!r}. Did the shell eat your angle brackets? "
              f'Use quotes like --boundary-token "{boundary_token}"', flush=True)

    with open(in_txt_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, max(1, workers), bt.encode("utf-8"))
    if len(boundaries) < 2:
        # No token? fallback to whole file
        print(f"[warning] boundary token {bt!r} not found in {in_txt_path.name}; "
              f"processing as a single chunk.", flush=True)
        return [0, in_txt_path.stat().st_size]
    return boundaries


def _parallel_encode_one_file(
    in_txt_path: Path,
    out_bin_path: Path,
    vocab_pkl: str,
    merges_pkl: str,
    special_tokens: List[str],
    np_dtype_str: str,
    num_workers: int,
    boundary_token: str,
    progress_every: int = 10,
) -> int:
    """
    Parallel-encode a single file to memmap-friendly .bin with progress logs.
    Returns total number of tokens written.
    """
    in_txt_path = Path(in_txt_path)
    out_bin_path = Path(out_bin_path)

    # 1) Boundaries
    workers = num_workers or os.cpu_count() or 1
    print(f"[boot] starting {in_txt_path.name} with {workers} workers", flush=True)
    t0 = time.time()
    boundaries = _find_boundaries_or_warn(in_txt_path, workers, boundary_token)
    n_chunks = len(boundaries) - 1
    print(f"[info] {in_txt_path.name}: {n_chunks} chunks", flush=True)

    # 2) Prepare tasks
    tasks = [
        (idx, str(in_txt_path), int(start), int(end), np_dtype_str)
        for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]

    # 3) Pool encode with progress
    pieces: List[Tuple[int, str, int]] = []
    done = 0
    last_log = time.time()

    from multiprocessing import Pool
    with Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(vocab_pkl, merges_pkl, special_tokens),
    ) as pool:
        for idx, tmp_path, n in pool.imap_unordered(_encode_chunk_worker, tasks, chunksize=1):
            pieces.append((idx, tmp_path, n))
            done += 1
            now = time.time()
            # Print either every N chunks or every ~2 seconds, whichever first
            if (done % progress_every == 0) or (now - last_log > 2) or (done == n_chunks):
                elapsed = now - t0
                pct = 100.0 * done / n_chunks if n_chunks else 100.0
                print(f"[progress] {in_txt_path.name}: {done}/{n_chunks} chunks "
                      f"({pct:.1f}%) in {elapsed:.1f}s", flush=True)
                last_log = now

    # 4) Concatenate shards
    print(f"[info] concatenating {len(pieces)} shards -> {out_bin_path.name}", flush=True)
    total_tokens = _concatenate_temp_bins(pieces, out_bin_path, np_dtype_str)

    elapsed = time.time() - t0
    rate = (total_tokens / elapsed) if elapsed > 0 else 0.0
    print(f"[done] {in_txt_path.name}: {total_tokens} tokens in {elapsed:.1f}s "
          f"({rate:.0f} tok/s)", flush=True)
    return total_tokens


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Parallel-encode TinyStories (train/val) with progress.")
    ap.add_argument("--vocab", required=True, help="Pickle: vocab (id -> bytes)")
    ap.add_argument("--merges", required=True, help="Pickle: merges (list of (bytes, bytes))")
    ap.add_argument("--train", required=True, help="Path to TinyStories train .txt")
    ap.add_argument("--val", required=True, help="Path to TinyStories val .txt")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--dtype", default="int32", choices=["int16", "int32", "int64"], help="Output dtype")
    ap.add_argument("--special", nargs="*", default=["<|endoftext|>"], help="Special tokens used during training")
    ap.add_argument("--boundary-token", default="<|endoftext|>", help="Token used for chunk boundaries")
    ap.add_argument("--workers", type=int, default=0, help="Number of worker processes (0 = CPU count)")
    ap.add_argument("--progress-every", type=int, default=10, help="Log every N chunks (also every ~2s)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # File 1: Train
    train_out = out_dir / "train.bin"
    print(f"[info] parallel-encoding train -> {train_out}", flush=True)
    n_train = _parallel_encode_one_file(
        in_txt_path=Path(args.train),
        out_bin_path=train_out,
        vocab_pkl=args.vocab,
        merges_pkl=args.merges,
        special_tokens=args.special,
        np_dtype_str=args.dtype,
        num_workers=args.workers,
        boundary_token=args.boundary_token,
        progress_every=args.progress_every,
    )
    _write_meta_json(train_out, n_train, args.dtype)
    print(f"[ok] train tokens: {n_train}  ->  {train_out}", flush=True)

    # File 2: Val
    val_out = out_dir / "val.bin"
    print(f"[info] parallel-encoding val -> {val_out}", flush=True)
    n_val = _parallel_encode_one_file(
        in_txt_path=Path(args.val),
        out_bin_path=val_out,
        vocab_pkl=args.vocab,
        merges_pkl=args.merges,
        special_tokens=args.special,
        np_dtype_str=args.dtype,
        num_workers=args.workers,
        boundary_token=args.boundary_token,
        progress_every=args.progress_every,
    )
    _write_meta_json(val_out, n_val, args.dtype)
    print(f"[ok] val tokens:   {n_val}  ->  {val_out}", flush=True)

    print("\nDone. Load later with:")
    print("  import json, numpy as np, pathlib")
    print(f"  p = pathlib.Path('{train_out}')")
    print("  meta = json.loads((p.with_suffix(p.suffix + '.meta.json')).read_text())")
    print("  arr = np.memmap(p, mode='r', dtype=np." + args.dtype + ", shape=(meta['length'],))", flush=True)


if __name__ == "__main__":
    main()
