from dataclasses import dataclass
from typing import Literal
from pathlib import Path
from cs336_basics.optimizer import optim_config
from cs336_basics.transformer import model_config
from cs336_basics.dataloader import data_config

@dataclass(frozen=False)
class TrainerConfig:
    load_from: str | None = None
    device: str = "mps"
    dtype: Literal["float32", "bfloat16"] = "float32"
    max_steps: int = 2000
    z_loss_weight: float = 1e-4
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    # run_name: str = "{date}_{optim.lr}"  # template
    run_name: str = "{date}"  # template
    save_dir: str | Path = Path(__file__).parent / "checkpoints"
    # save every n steps
    save_every: int = 100
    # validate every n steps
    val_every: int = 100
    # log train metrics every n steps
    log_every: int = 10
    
    warmup_steps: int = 2_000                   # linear warmup to base LR
    lr_decay_style: Literal["cosine","linear","constant"] = "cosine"
    min_lr_factor: float = 0.10        

@dataclass(frozen=False)
class Config:
    data: data_config.MMDatasetConfig
    model: model_config.TransformerConfig
    optim: optim_config.AnyOptimConfig
    trainer: TrainerConfig
    project: str = "cs336"  # for wandb

default_cfg = Config(data_config.MMDatasetConfig(),
                      model_config.TransformerConfig(),
                        optim_config.AdamWFactory(),
                          TrainerConfig())    

default_cuda_cfg = Config(data_config.MMDatasetConfig(context_length=512,batch_size=64, val_batch_size=64),
                      model_config.TransformerConfig(context_length=512),
                        optim_config.AdamWFactory(),
                          TrainerConfig(device="cuda", dtype="bfloat16"))   
