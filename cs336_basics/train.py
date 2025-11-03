#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from dataclasses import asdict, replace
from datetime import datetime
from typing import Any
# --- your package imports ---
from cs336_basics.training.config import default_cuda_cfg as base_cfg
from cs336_basics.training.trainer import Trainer
from cs336_basics.training.config import Config

from cs336_basics.training import config as training_config  # for type hints if you want: training_config.Config
# from cs336_basics.utils.config_tools import apply_overrides, dataclass_to_nested_dict, wandb_run_name
from cs336_basics.logger import logger  # matches your Trainer's logger import


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """
    Flatten the dict
    """
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out

def render_template(template: str, cfg, extra: dict[str, Any] | None = None) -> str:
    base = flatten(dataclass_to_nested_dict(cfg))
    base["date"] = datetime.now().strftime("%m%d")
    base["time"] = datetime.now().strftime("%H%M")
    if extra:
        base.update(extra)

    class DotDict(dict):
        def __missing__(self, key):
            return "NA"

    return template.format_map(base)


def wandb_run_name(cfg: Config, extra: dict[str, Any] | None = None) -> str:
    return render_template(cfg.trainer.run_name, cfg, extra)

# (Optional) wandb is purely best-effort; your Trainer doesn't require it.
try:
    import wandb
    _HAVE_WANDB = True
except Exception:
    _HAVE_WANDB = False


def dataclass_to_nested_dict(dc) -> dict[str, Any]:
    return asdict(dc)



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-path",
        type=str,
        default=str(Path(__file__).parent.parent / "tokenized_parallel/train.bin"),
    )
    p.add_argument(
        "--validation-path",
        type=str,
        default=str(Path(__file__).parent.parent / "tokenized_parallel/val.bin"),
    )
    p.add_argument(
        "--override",
        type=str,
        help='JSON string of overrides, e.g. \'{"trainer.max_steps": 2000, "trainer.device": "cuda"}\'',
    )
    p.add_argument(
        "--use-wandb",
        action="store_true",
        help="If set (and wandb is installed), log a run with the final resolved config.",
    )
    return p.parse_args()


def _suffix_from_override(override_dict: dict) -> str:
    """
    Build a short suffix for run_name/save_dir from override keys/values,
    similar to your example. Keeps it stable and compact.
    """
    if not override_dict:
        return ""
    parts = []
    for k, v in override_dict.items():
        parts.append(f"{k.split('.')[-1][:4]}={str(v)[:4]}")
    return "_".join(parts)


def train(cfg: "training_config.Config", args):
    # Parse user overrides
    user_override = json.loads(args.override) if args.override else {}

    # Compose a short suffix for the run based on overrides
    suffix = _suffix_from_override(user_override)

    # Prepare run name (purely for logging; Trainer uses cfg.trainer.run_name internally)
    try:
        base_run_name = wandb_run_name(cfg)
    except Exception:
        # Fallback if wandb_run_name isn't available for some reason
        base_run_name = "run"
    run_name = f"{base_run_name}_{suffix}" if suffix else base_run_name
    run_name = run_name[:64]  # keep it tidy

    logger.info(f"Training run: {run_name}")

    # Always ensure data paths come from CLI unless overridden explicitly
    # merged_override = {
    #     **user_override,
    #     "data.train_path": args.train_path,
    #     "data.validation_path": args.validation_path,
    # }

    # Apply overrides to the base config
    # cfg = apply_overrides(cfg, merged_override)

    # Optionally tweak save_dir by suffix so parallel runs don't collide
    if suffix:
        try:
            # If save_dir is already a Path in your config, this will work;
            # if it's a string, Path() will wrap it.
            cfg.trainer.save_dir = Path(cfg.trainer.save_dir) / suffix
        except AttributeError:
            # If cfg.trainer.save_dir isn't present for some reason, ignore gracefully
            pass

    # Best-effort wandb init (does not change Trainer behavior)
    run = None
    if args.use_wandb and _HAVE_WANDB:
        run = wandb.init(
            project=getattr(cfg, "project", "cs336_basics"),
            name=run_name,
            config=dataclass_to_nested_dict(cfg),
        )
        # No need to pass into Trainer; your Trainer handles logging internally.

    # Build trainer and run training
    trainer = Trainer(cfg)
    trainer.train()

    # Finish wandb if we started it
    if run is not None:
        run.finish()


if __name__ == "__main__":
    train(base_cfg, parse_args())
