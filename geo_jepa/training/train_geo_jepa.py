#!/usr/bin/env python3
"""
Geo-JEPA Distributed Training Script.

Orchestrates co-training with:
- Flow-Matching action loss (L_FM)
- Semantic world-model loss (beta * L_WM_sem)
- Geometric world-model loss (gamma * L_WM_geo)
- Current-frame geometric alignment loss (alpha * L_geo)
- Progressive linear/cosine warmup schedules for alpha and gamma.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm

import sys
sys.path.insert(0, "/home/kavinder/Geo-JEPA")
sys.path.insert(0, "/home/kavinder/geo-jepa-dev/VLA-JEPA")

from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, normalize_dotlist_args
from geo_jepa.models.geo_jepa_framework import Geo_JEPA
from geo_jepa.training.warmup_scheduler import GeoJEPALossScheduler

logger = get_logger(__name__)


def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")
    return output_dir


def train_geo_jepa(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    accelerator = Accelerator()
    set_seed(42)

    output_dir = setup_directories(cfg)

    # Instantiate Geo-JEPA Framework
    logger.info(f"Initializing Geo-JEPA framework...")
    model = Geo_JEPA(config=cfg)

    # Initialize loss coefficient warmup scheduler
    geo_cfg = getattr(cfg.framework, "geometric_forcing", {})
    loss_scheduler = GeoJEPALossScheduler(
        alpha_target=geo_cfg.get("alpha", 0.5),
        alpha_warmup_steps=geo_cfg.get("warmup_steps", 2000),
        alpha_schedule=geo_cfg.get("warmup_schedule", "linear"),
        beta=cfg.framework.vj2_model.get("beta", 0.1),
        gamma_target=geo_cfg.get("gamma", 0.1),
        gamma_warmup_steps=geo_cfg.get("warmup_steps", 2000),
        gamma_schedule=geo_cfg.get("warmup_schedule", "linear")
    )

    # Setup Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas)
    )

    model, optimizer = accelerator.prepare(model, optimizer)
    logger.info(f"Geo-JEPA model prepared for training with {cfg.trainer.max_train_steps} steps.")

    return model, loss_scheduler, optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geo-JEPA Training Launcher")
    parser.add_argument("--config", type=str, default="/home/kavinder/Geo-JEPA/configs/full_geo_jepa.yaml")
    args = parser.parse_args()
    print(f"[Geo-JEPA Trainer] Loaded config: {args.config}")
    model, loss_sched, opt = train_geo_jepa(args.config)
    print("[Geo-JEPA Trainer] Model and loss scheduler initialized successfully.")
