import os
import math
from pathlib import Path
from datetime import datetime

import torch
from cs336_basics.training import config
from cs336_basics.optimizer import optim_config
from cs336_basics.transformer import model_config
from cs336_basics.dataloader import data_config
from cs336_basics.training.losses import cross_entropy_with_unweighted_zloss
from cs336_basics.logger import logger


class Trainer:
    def __init__(self, cfg: config.Config):
        self.cfg = cfg

        # Build model → data → optimizer (in that order)
        if cfg.trainer.dtype == 'bfloat16' or cfg.trainer.dtype == 'bf16':
            dtype = torch.bfloat16
        elif cfg.trainer.dtype == 'float32':
            dtype = torch.float32
            
        device = torch.device(cfg.trainer.device)
        self.model = cfg.model.make(device, dtype)
        self.train_data, self.val_data = cfg.data.make()
        self.optimizer = cfg.optim.make(self.model.parameters())

        # AMP / device
        self._device_type = "cuda" if "cuda" in cfg.trainer.device else ("mps" if "mps" in cfg.trainer.device else "cpu")

        if cfg.trainer.dtype in ("bfloat16", "float16") and self._device_type != "cuda":
            # Downgrade to float32 on non-CUDA
            from cs336_basics.logger import logger
            logger.warning(f"{self._device_type} does not fully support {cfg.trainer.dtype} for training; using float32.")
            self.autocast_dtype = torch.float32
            self.use_amp = False
            self.scaler = None
        else:
            if cfg.trainer.dtype == "bfloat16":
                self.autocast_dtype = torch.bfloat16
                self.use_amp = True
                self.scaler = None
            else:
                self.autocast_dtype = torch.float32
                self.use_amp = False
                self.scaler = None

        # LR Scheduler setup
        self._setup_scheduler()
        self.z_weight = cfg.trainer.z_loss_weight

        # Checkpoint/run folder
        self.global_step = 0
        self.best_val = float("inf")
        self.run_dir = self._init_run_dir()

        # Optional resume
        if cfg.trainer.load_from:
            self._load_checkpoint(cfg.trainer.load_from)

    # --------------------------
    # Scheduler helpers
    # --------------------------
    def _setup_scheduler(self):
        """Capture each group's base LR so we can scale them consistently."""
        self._base_lrs = [g.get("lr", 0.0) for g in self.optimizer.param_groups]
        if any(lr is None for lr in self._base_lrs):
            self._base_lrs = [float(g["lr"]) for g in self.optimizer.param_groups]

        self._warmup_steps = max(0, int(self.cfg.trainer.warmup_steps))
        self._total_steps = max(1, int(self.cfg.trainer.max_steps))
        self._decay_steps = max(1, self._total_steps - self._warmup_steps)
        self._min_factor = float(self.cfg.trainer.min_lr_factor)

    def _lr_multiplier(self, step: int) -> float:
        """Warmup then decay (cosine/linear/constant)."""
        if step < self._warmup_steps and self._warmup_steps > 0:
            return (step + 1) / self._warmup_steps  # avoid 0 on first step

        progress = min(1.0, max(0.0, (step - self._warmup_steps) / self._decay_steps))
        style = self.cfg.trainer.lr_decay_style
        if style == "cosine":
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 → 0
            return self._min_factor + (1.0 - self._min_factor) * cos
        elif style == "linear":
            return max(self._min_factor, 1.0 - progress * (1.0 - self._min_factor))
        else:  # "constant"
            return 1.0

    def _set_lr(self, step: int) -> float:
        mult = self._lr_multiplier(step)
        for base_lr, group in zip(self._base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * mult
        return float(self.optimizer.param_groups[0]["lr"])  # for logging

    # --------------------------
    # Loss helper (handles 2D or 3D logits)
    # --------------------------
    @staticmethod
    def _compute_ce_and_z(logits: torch.Tensor, targets: torch.Tensor):
        if logits.dim() == 3:
            B, T, V = logits.shape
            logits = logits.reshape(B * T, V)
            targets = targets.reshape(B * T)

        # Ensure dtypes are loss-friendly
        logits = logits.float()           # force fp32 for CPU/MPS stability
        if targets.dtype != torch.long:
            targets = targets.long()

        return cross_entropy_with_unweighted_zloss(logits, targets)

    # --------------------------
    # One training step
    # --------------------------
    def step(self, inputs, targets, step_idx: int):
        self.model.train()
        cur_lr = self._set_lr(step_idx)
        self.optimizer.zero_grad(set_to_none=True)

        autocast_enabled = self.use_amp and self._device_type in {"cuda", "cpu", "mps"}
        with torch.autocast(device_type=self._device_type,
                            dtype=self.autocast_dtype,
                            enabled=autocast_enabled):
            logits = self.model(inputs)
            ce_loss, z_loss = self._compute_ce_and_z(logits, targets)
            loss = ce_loss + self.z_weight * z_loss

        loss.backward()

        grad_norm = None
        if self.cfg.trainer.max_grad_norm is not None:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.trainer.max_grad_norm
                ).item()
            )

        self.optimizer.step()

        return {
            "train_loss": float(loss.detach().item()),
            "train_ce": float(ce_loss.detach().item()),
            "z_loss_unweighted": float(z_loss.detach().item()),
            "grad_norm": grad_norm,
            "lr": cur_lr,
        }

    # --------------------------
    # Validation
    # --------------------------
    @torch.no_grad()
    def evaluate(self, max_batches: int = 10):
        """Evaluate on a small number of validation batches to keep it cheap."""
        self.model.eval()
        total_ce, total_z, n = 0.0, 0.0, 0

        it = 0
        for inputs, targets in self.val_data.get_batch_iterator(
            self.cfg.data.val_batch_size, shuffle=False
        ):
            logits = self.model(inputs)
            ce, z = self._compute_ce_and_z(logits, targets)
            total_ce += float(ce.item())
            total_z += float(z.item())
            it += 1
            if it >= max_batches:
                break

        if it == 0:
            return {"val_ce": float("nan"), "val_z": float("nan"), "val_loss": float("nan")}
        val_ce = total_ce / it
        val_z = total_z / it
        val_loss = val_ce + self.z_weight * val_z
        return {"val_ce": val_ce, "val_z": val_z, "val_loss": val_loss}

    # --------------------------
    # Checkpointing
    # --------------------------
    def _init_run_dir(self) -> Path:
        save_root = Path(self.cfg.trainer.save_dir)
        save_root.mkdir(parents=True, exist_ok=True)
        # Fill {date} token if present
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = self.cfg.trainer.run_name.format(date=date_str)
        run_dir = save_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _checkpoint_payload(self, step: int) -> dict:
        return {
            "step": step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "base_lrs": self._base_lrs,
            "best_val": self.best_val,
            "config": self.cfg,  # dataclass is picklable; fine for local use
        }

    def _save_checkpoint(self, step: int, tag: str | None = None):
        ckpt_name = f"step_{step:06d}.pt" if tag is None else f"{tag}.pt"
        path = self.run_dir / ckpt_name
        torch.save(self._checkpoint_payload(step), path)
        # also save/update "last.pt"
        torch.save(self._checkpoint_payload(step), self.run_dir / "last.pt")
        logger.info(f"Saved checkpoint to {path}")

    def _save_best(self, step: int):
        torch.save(self._checkpoint_payload(step), self.run_dir / "best.pt")
        logger.info(f"📌 New best validation; saved best.pt at step {step}")

    def _load_checkpoint(self, path: str | os.PathLike):
        path = Path(path)
        logger.info(f"Loading checkpoint from {path}")
        payload = torch.load(path, map_location=self.cfg.trainer.device)
        self.model.load_state_dict(payload["model_state"])
        try:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {e}")
        self.global_step = int(payload.get("step", 0))
        self._base_lrs = payload.get("base_lrs", self._base_lrs)
        self.best_val = float(payload.get("best_val", float("inf")))
        logger.info(f"Resumed at step={self.global_step}, best_val={self.best_val:.4f}")

    # --------------------------
    # Training loop
    # --------------------------
    def train(self):
        logger.info("Training started.")
        logger.info(f"Config: {self.cfg}")

        # Basic loop over batches. Uses dataset's own iterator.
        train_iter = self.train_data.get_batch_iterator(self.cfg.data.batch_size, shuffle=True)

        for step in range(self.global_step, self.cfg.trainer.max_steps):
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                # re-create a fresh shuffled iterator
                train_iter = self.train_data.get_batch_iterator(self.cfg.data.batch_size, shuffle=True)
                inputs, targets = next(train_iter)

            metrics = self.step(inputs, targets, step)
            self.global_step = step + 1

            # Logging
            if step % self.cfg.trainer.log_every == 0:
                logger.info(
                    f"step={step:6d} "
                    f"lr={metrics['lr']:.6g} "
                    f"loss={metrics['train_loss']:.4f} "
                    f"ce={metrics['train_ce']:.4f} "
                    f"z={metrics['z_loss_unweighted']:.4f} "
                    f"gn={0.0 if metrics['grad_norm'] is None else metrics['grad_norm']:.4f}"
                )

            # Validation
            if step % self.cfg.trainer.val_every == 0 and step > 0:
                val = self.evaluate(max_batches=10)
                logger.info(
                    f"[val] step={step:6d} "
                    f"val_loss={val['val_loss']:.4f} "
                    f"val_ce={val['val_ce']:.4f} "
                    f"val_z={val['val_z']:.4f}"
                )
                # Track and save best model
                if val["val_loss"] < self.best_val:
                    self.best_val = val["val_loss"]
                    self._save_best(step)

            # Save periodic checkpoints
            if step % self.cfg.trainer.save_every == 0 and step > 0:
                self._save_checkpoint(step)

        # Final save at end
        self._save_checkpoint(self.global_step, tag="final")
        logger.info("Training complete.")
