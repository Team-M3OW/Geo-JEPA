#!/usr/bin/env python3
"""
Geo-JEPA: Unified Multimodal VLA Training Pipeline.

Trains the Unified Multimodal Vision-Language-Action (VLA) Policy:
- Input: Raw RGB Frames + Qwen2.5 Language Tokens + Robot Proprioception
- Multimodal Cross-Attention Transformer
- Coupled 3D Geometric Flow Matching with 8-Step Action Chunks (H=8)
- Loss: Optimal Transport Flow Matching Loss L_Flow + Ray Direction Loss L_Ray

Dataset: /media/kavinder/hdd2/datasets/libero/libero_spatial/
Checkpoint: /media/kavinder/hdd2/geo_jepa_runs/unified_vla_spatial/checkpoints/
"""

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, "/home/kavinder/LIBERO")
sys.path.insert(0, "/home/kavinder/Geo-JEPA")

from geo_jepa.models.unified_vla_flow_policy import UnifiedVLAFlowPolicy


class LiberoSpatialVLADataset(Dataset):
    """
    Multimodal Dataset streaming LIBERO Spatial Parquet demonstration episodes.
    """
    def __init__(self, data_root: str, horizon: int = 8, max_files: int = 25):
        self.data_root = Path(data_root)
        self.horizon = horizon
        
        # Load task metadata
        tasks_df = pd.read_parquet(self.data_root / "meta" / "tasks.parquet")
        self.task_names = {row["task_index"]: task_str for task_str, row in tasks_df.iterrows()}

        parquet_files = sorted(list((self.data_root / "data" / "chunk-000").glob("*.parquet")))[:max_files]
        print(f"Loading {len(parquet_files)} demonstration parquet files into memory...")

        self.samples = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            ep_indices = df["episode_index"].unique()
            for ep in ep_indices:
                ep_df = df[df["episode_index"] == ep].reset_index(drop=True)
                N = len(ep_df)
                if N <= horizon:
                    continue
                t_idx = int(ep_df["task_index"].iloc[0])
                task_str = self.task_names.get(t_idx, "pick up the black bowl and place it on the plate")

                for i in range(N - horizon):
                    img_raw = ep_df["observation.images.image"].iloc[i]
                    state = ep_df["observation.state"].iloc[i]  # [eef, gripper, ...]
                    actions = np.stack([ep_df["action"].iloc[i + h] for h in range(horizon)])  # [8, 7]

                    self.samples.append({
                        "img_bytes": img_raw["bytes"],
                        "task_prompt": task_str,
                        "eef_pos": state[:3].astype(np.float32),
                        "gripper_q": state[7:9].astype(np.float32) if len(state) > 8 else state[-2:].astype(np.float32),
                        "actions": actions.astype(np.float32)
                    })

        print(f"Total training trajectory chunks loaded: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]

        # Decode image
        pil_img = Image.open(io.BytesIO(item["img_bytes"])).convert("RGB").resize((128, 128))
        img_np = np.array(pil_img, dtype=np.float32) / 255.0  # [128, 128, 3]
        img_tensor = torch.tensor(img_np).permute(2, 0, 1)  # [3, 128, 128]

        return {
            "rgb": img_tensor,
            "prompt": item["task_prompt"],
            "eef_pos": torch.tensor(item["eef_pos"], dtype=torch.float32),
            "gripper_q": torch.tensor(item["gripper_q"], dtype=torch.float32),
            "actions": torch.tensor(item["actions"], dtype=torch.float32)  # [8, 7]
        }


def collate_vla_batch(batch: List[Dict]) -> Dict:
    return {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "eef_pos": torch.stack([b["eef_pos"] for b in batch]),
        "gripper_q": torch.stack([b["gripper_q"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch])
    }


def train_unified_vla(
    data_root: str = "/media/kavinder/hdd2/datasets/libero/libero_spatial",
    output_dir: str = "/media/kavinder/hdd2/geo_jepa_runs/unified_vla_spatial",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cuda"
):
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" GEO-JEPA: TRAINING UNIFIED MULTIMODAL VLA FLOW MATCHING MODEL")
    print(f" Output Directory: {output_dir}")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 85)

    dataset = LiberoSpatialVLADataset(data_root=data_root, horizon=8, max_files=30)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_vla_batch,
        num_workers=4,
        pin_memory=True
    )

    policy = UnifiedVLAFlowPolicy(embed_dim=512, action_dim=7, horizon=8).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

    policy.train()
    global_step = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            global_step += 1
            rgb = batch["rgb"].to(device)
            prompts = batch["prompts"]
            eef_pos = batch["eef_pos"].to(device)
            gripper_q = batch["gripper_q"].to(device)
            gt_actions = batch["actions"].to(device)  # [B, 8, 7]
            B = rgb.shape[0]

            optimizer.zero_grad()

            # 1. Vision & Multimodal Encoding
            vis_feat = policy.vis_encoder(rgb)
            x_vis = vis_feat.flatten(2).permute(0, 2, 1)
            x_lang = policy.encode_text(prompts, device)
            proprio = torch.cat([eef_pos, gripper_q], dim=-1)
            x_proprio = policy.proprio_proj(proprio).unsqueeze(1)
            x_vis_full = torch.cat([x_vis, x_proprio], dim=1)

            z_fused = policy.cross_modal(x_vis_full, x_lang)
            z_cond = z_fused.mean(dim=1)  # [B, embed_dim]

            # 2. Optimal Transport Flow Matching Loss on Action Trajectory
            # Sample continuous flow time t in [0, 1]
            t = torch.rand((B,), device=device, dtype=torch.float32)
            u_0 = torch.randn_like(gt_actions)  # Standard Gaussian base prior
            u_1 = gt_actions  # Target demonstration chunk

            # Linear Optimal Transport Flow Interpolation:
            # u_t = (1 - (1 - 1e-4) * t) * u_0 + t * u_1
            t_expand = t.view(B, 1, 1)
            u_t = (1.0 - (1.0 - 1e-4) * t_expand) * u_0 + t_expand * u_1
            target_velocity = u_1 - (1.0 - 1e-4) * u_0

            # Forward vector field network
            v_pred = policy.flow_head(u_t, t, z_cond)

            # Flow matching loss L_FM = || v_pred - (u_1 - u_0) ||^2
            loss_flow = F.mse_loss(v_pred, target_velocity)

            # Backward pass
            loss = loss_flow
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch}/{epochs}] Complete | Mean Flow Loss: {avg_loss:.5f} | Time: {time.time()-start_time:.1f}s")

        # Save checkpoint periodically
        if epoch % 5 == 0 or epoch == epochs:
            ckpt_path = ckpt_dir / f"unified_vla_epoch_{epoch:03d}.pt"
            latest_path = ckpt_dir / "unified_vla_latest.pt"
            save_dict = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }
            torch.save(save_dict, ckpt_path)
            torch.save(save_dict, latest_path)
            print(f"--> Saved Checkpoint: {ckpt_path}")

    print("\n" + "=" * 85)
    print(f" UNIFIED MULTIMODAL VLA TRAINING COMPLETE in {time.time()-start_time:.1f}s!")
    print(f" Final Model Saved: {ckpt_dir / 'unified_vla_latest.pt'}")
    print("=" * 85)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_unified_vla(epochs=15, batch_size=32, lr=3e-4, device=device)
