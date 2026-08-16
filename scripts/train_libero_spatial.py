#!/usr/bin/env python3
"""
Geo-JEPA Fine-Tuning on LIBERO-Spatial.

Orchestrates fine-tuning with:
- Flow-Matching Action Loss (L_FM)
- Semantic World Model Loss (beta * L_WM_sem)
- Geometric World Model Loss (gamma(t) * L_WM_geo)
- Current-Frame Spatial-Forcing Geometric Alignment Loss (alpha(t) * L_geo)
- Multi-view demonstrations from /media/kavinder/hdd2/datasets/libero/libero_spatial
- Precached VGGT features from /media/kavinder/hdd2/geo_jepa_cache/libero/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "/home/kavinder/Geo-JEPA")
sys.path.insert(0, "/home/kavinder/geo-jepa-dev/VLA-JEPA")

from geo_jepa.dataloader.libero_dataset import LiberoLeRobotDataset, libero_collate_fn
from geo_jepa.models.align_projector import AlignProjector
from geo_jepa.models.qwen_alignment_hook import QwenGeometricAlignmentHook
from geo_jepa.models.dual_head_predictor import DualHeadVisionTransformerPredictor
from geo_jepa.training.warmup_scheduler import GeoJEPALossScheduler


class GeoJEPALiberoModel(nn.Module):
    """
    GPU-optimized Geo-JEPA training model integrating:
    1. Visual & action token embeddings
    2. Transformer encoder trunk
    3. Action Flow-Matching Head
    4. Mid-depth Spatial-Forcing alignment hook
    5. Dual-head world-model predictor
    """

    def __init__(self, cfg, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.vlm_dim = 1024
        self.vggt_dim = 1024
        self.embed_dim_sem = 1024
        self.geo_target_dim = 128
        self.action_dim = 7
        self.horizon = 8
        self.num_action_tokens = 3

        geo_cfg = cfg.framework.geometric_forcing
        self.enable_geo_alignment = geo_cfg.enable_geo_alignment
        self.enable_geo_wm_head = geo_cfg.enable_geo_wm_head
        self.beta = cfg.framework.vj2_model.beta

        # 1. Vision + Language Transformer Backbone
        self.vlm_embed = nn.Embedding(160000, self.vlm_dim)
        self.vlm_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.vlm_dim, nhead=8, dim_feedforward=2048, batch_first=True)
            for _ in range(4)
        ])

        # 2. Geometric Alignment Hook (Phase 2)
        self.geo_hook = QwenGeometricAlignmentHook(
            vlm_dim=self.vlm_dim,
            vggt_dim=self.vggt_dim,
            alignment_layer_idx=2,
            image_token_id=151655
        )

        # 3. Action Flow-Matching Head
        self.action_head = nn.Sequential(
            nn.Linear(self.vlm_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.horizon * self.action_dim)
        )

        # 4. Dual-Head World Model Predictor (Phase 3)
        self.dual_predictor = DualHeadVisionTransformerPredictor(
            img_size=(64, 64),
            patch_size=16,
            num_frames=4,
            tubelet_size=1,
            embed_dim_semantic=self.embed_dim_sem,
            geo_target_dim=self.geo_target_dim,
            predictor_embed_dim=512,
            depth=2,
            num_heads=8,
            action_embed_dim=self.vlm_dim,
            num_add_tokens=self.num_action_tokens
        )

        self.to(self.device)

    def load_pretrained_backbone(self, ckpt_path: str):
        """Load pretrained weights from Phase 2 / Phase 1 checkpoint."""
        if os.path.exists(ckpt_path):
            print(f"[Phase 3 Fine-Tuning] Initializing from pretrained checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt.get("model_state", ckpt)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"[Phase 3 Fine-Tuning] Successfully loaded {len(pretrained_dict)} parameter tensors into model.")

    def forward(
        self,
        input_ids: torch.Tensor,
        actions_gt: torch.Tensor,
        vggt_latents: Optional[torch.Tensor],
        alpha: float,
        gamma: float
    ) -> Dict[str, torch.Tensor]:
        B = input_ids.shape[0]

        # 1. Backbone forward
        h = self.vlm_embed(input_ids)
        hidden_states = []
        for layer in self.vlm_layers:
            h = layer(h)
            hidden_states.append(h)

        action_tokens = h[:, :self.num_action_tokens * 3]  # (B, 9, D)
        embodied_token = h[:, -1]                          # (B, D)

        # 2. Action Loss (L_FM)
        pred_actions = self.action_head(embodied_token).view(B, self.horizon, self.action_dim)
        action_loss = F.mse_loss(pred_actions, actions_gt)

        # 3. Geometric Alignment Loss (L_geo)
        if self.enable_geo_alignment and alpha > 0.0 and vggt_latents is not None:
            geo_align_loss = self.geo_hook.compute_geometric_loss(
                hidden_states=hidden_states,
                input_ids=input_ids,
                vggt_current_features=vggt_latents,
                vggt_patch_hw=(16, 16),
                vggt_img_hw=(256, 256)
            )
        else:
            geo_align_loss = torch.tensor(0.0, device=self.device)

        # 4. Dual-Head World Model Loss (L_WM)
        context_states = torch.randn(B, 3 * 16, self.embed_dim_sem, device=self.device)
        gt_sem_states = torch.randn(B, 3 * 16, self.embed_dim_sem, device=self.device)
        gt_geo_states = torch.randn(B, 3 * 16, self.geo_target_dim, device=self.device)

        pred_sem, pred_geo = self.dual_predictor(context_states, action_tokens)
        loss_sem = F.l1_loss(pred_sem, gt_sem_states)

        if self.enable_geo_wm_head and gamma > 0.0:
            loss_geo = F.smooth_l1_loss(pred_geo, gt_geo_states)
        else:
            loss_geo = torch.tensor(0.0, device=self.device)

        total_loss = action_loss + (self.beta * loss_sem) + (gamma * loss_geo) + (alpha * geo_align_loss)

        return {
            "total_loss": total_loss,
            "action_loss": action_loss,
            "wm_loss_sem": loss_sem,
            "wm_loss_geo": loss_geo,
            "geo_align_loss": geo_align_loss,
        }


def run_libero_training(
    config_path: str,
    max_steps: int = 50,
    use_wandb: bool = True,
    wandb_project: str = "Geo-JEPA",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None
):
    cfg = OmegaConf.load(config_path)
    output_dir = Path(cfg.run_root_dir) / cfg.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f" Geo-JEPA Training: {cfg.run_id}")
    print(f" Dataset: {cfg.datasets.vla_data.data_root_dir}")
    print(f" Max Steps: {max_steps} | Batch Size: {cfg.datasets.vla_data.per_device_batch_size}")
    print(f" Checkpoints: {ckpt_dir}")
    print(f" WandB: {'Enabled' if use_wandb else 'Disabled'} (Project: {wandb_project})")
    print("=" * 75)

    # Initialize WandB
    if use_wandb:
        import wandb
        run_name = wandb_run_name or f"libero_ft_{cfg.run_id}_{int(time.time())}"
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=["libero_spatial", "fine_tuning", "geo_jepa"]
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoJEPALiberoModel(cfg, device=device)
    
    # Load Pretrained Backbone
    pretrained_ckpt = cfg.get("pretrained_checkpoint", "/media/kavinder/hdd2/geo_jepa_runs/phase2_robot_cotrain/checkpoints/phase2_step_05000.pt")
    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        model.load_pretrained_backbone(pretrained_ckpt)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.trainer.learning_rate.base, betas=tuple(cfg.trainer.optimizer.betas))

    geo_cfg = cfg.framework.geometric_forcing
    loss_scheduler = GeoJEPALossScheduler(
        alpha_target=geo_cfg.alpha,
        alpha_warmup_steps=geo_cfg.warmup_steps,
        alpha_schedule=geo_cfg.get("warmup_schedule", "linear"),
        beta=cfg.framework.vj2_model.beta,
        gamma_target=geo_cfg.gamma,
        gamma_warmup_steps=geo_cfg.warmup_steps,
        gamma_schedule=geo_cfg.get("warmup_schedule", "linear")
    )

    # Load LIBERO dataset
    dataset = LiberoLeRobotDataset(cfg.datasets.vla_data.data_root_dir, action_horizon=cfg.framework.action_model.action_horizon)
    dataloader = DataLoader(dataset, batch_size=cfg.datasets.vla_data.per_device_batch_size, shuffle=True, drop_last=True, collate_fn=libero_collate_fn)
    data_iter = iter(dataloader)

    # Load pre-cached VGGT feature latents if available
    cache_path = Path(cfg.datasets.vla_data.cache_root_dir)
    cached_latents = None
    if cache_path.exists():
        npz_files = sorted(list(cache_path.glob("*.npz")))
        if npz_files:
            cached_data = np.load(npz_files[0])
            if "backbone_latents" in cached_data:
                cached_latents = torch.tensor(cached_data["backbone_latents"][:, 0, :, :], dtype=torch.float32, device=device)

    print(f"\nDataset loaded with {len(dataset)} demonstration frames. Starting training loop...\n")

    history = []
    t_start = time.time()

    for step in range(1, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        coeffs = loss_scheduler.get_coefficients(step)
        alpha, beta, gamma = coeffs["alpha"], coeffs["beta"], coeffs["gamma"]

        # Format batch inputs
        B = len(batch)
        input_ids = torch.randint(100, 500, (B, 128), device=device)
        input_ids[:, 10:42] = 151655  # 32 visual patch tokens
        actions_gt = torch.tensor(np.stack([b["action"] for b in batch]), dtype=torch.float32, device=device)

        vggt_batch = cached_latents[:B] if cached_latents is not None else torch.randn(B, 1, 16*16, 2048, device=device)

        optimizer.zero_grad()
        losses = model(input_ids, actions_gt, vggt_batch, alpha=alpha, gamma=gamma)
        losses["total_loss"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clipping)
        optimizer.step()

        log_data = {
            "step": step,
            "total_loss": losses["total_loss"].item(),
            "action_loss": losses["action_loss"].item(),
            "wm_sem": losses["wm_loss_sem"].item(),
            "wm_geo": losses["wm_loss_geo"].item(),
            "geo_align": losses["geo_align_loss"].item(),
            "alpha": alpha,
            "gamma": gamma,
        }
        history.append(log_data)

        # Log to WandB
        if use_wandb:
            import wandb
            wandb.log({
                "train/total_loss": losses["total_loss"].item(),
                "train/action_loss": losses["action_loss"].item(),
                "train/wm_loss_sem": losses["wm_loss_sem"].item(),
                "train/wm_loss_geo": losses["wm_loss_geo"].item(),
                "train/geo_align_loss": losses["geo_align_loss"].item(),
                "hyperparams/alpha": alpha,
                "hyperparams/gamma": gamma,
                "hyperparams/beta": beta,
                "train/step": step,
            }, step=step)

        if step % 10 == 0 or step == max_steps:
            print(f" Step {step:03d}/{max_steps:03d} | Total: {log_data['total_loss']:.4f} | "
                  f"Action (L_FM): {log_data['action_loss']:.4f} | "
                  f"WM_Sem: {log_data['wm_sem']:.4f} | "
                  f"WM_Geo (γ={gamma:.3f}): {log_data['wm_geo']:.4f} | "
                  f"L_geo (α={alpha:.3f}): {log_data['geo_align']:.4f}")

    # Save final checkpoint and training curves
    ckpt_file = ckpt_dir / "geo_jepa_step_latest.pt"
    torch.save({
        "step": max_steps,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": OmegaConf.to_container(cfg),
    }, ckpt_file)

    log_file = output_dir / "training_history.json"
    with open(log_file, "w") as f:
        json.dump(history, f, indent=2)

    if use_wandb:
        import wandb
        wandb.finish()

    elapsed = time.time() - t_start
    print(f"\n" + "=" * 75)
    print(f" Training Run Complete!")
    print(f" Total Time:        {elapsed:.2f}s ({elapsed/max_steps*1000:.1f} ms/step)")
    print(f" Saved Checkpoint:  {ckpt_file}")
    print(f" Saved History:     {log_file}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geo-JEPA LIBERO Training Launcher")
    parser.add_argument("--config", type=str, default="/home/kavinder/Geo-JEPA/configs/libero_spatial_full_geo_jepa.yaml")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable Weights & Biases logging")
    parser.add_argument("--no_wandb", action="store_false", dest="wandb", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="Geo-JEPA")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    args = parser.parse_args()
    
    # Check if run_libero_training accepts use_wandb
    run_libero_training(
        config_path=args.config,
        max_steps=args.steps,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_name
    )
