#!/usr/bin/env python3
"""
Geo-JEPA: DINOv2-Grounded Multimodal Action Chunking Policy (ACT) Training Pipeline.

Integrates:
1. Pretrained DINOv2 Visual Tokenizer
2. Per-Dimension Action Space Normalization (mu, sigma)
3. Direct Multimodal Action Chunking Transformer (H=8)

Dataset: /media/kavinder/hdd2/datasets/libero/
Checkpoints: /media/kavinder/hdd2/geo_jepa_runs/dinov2_act_policy/checkpoints/
"""

import io
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, "/home/kavinder/LIBERO")
sys.path.insert(0, "/home/kavinder/Geo-JEPA")

from geo_jepa.models.dinov2_act_vla_policy import DINOv2ACTPolicy


class DINOv2LiberoDataset(Dataset):
    """
    Multimodal Dataset with DINOv2 Image Preprocessing & Action Normalization.
    """
    def __init__(self, data_root: str, horizon: int = 8, max_files: int = 40, augment: bool = True):
        self.data_root = Path(data_root)
        self.horizon = horizon
        self.augment = augment
        
        # Load task metadata
        tasks_df = pd.read_parquet(self.data_root / "meta" / "tasks.parquet")
        self.task_names = {row["task_index"]: task_str for task_str, row in tasks_df.iterrows()}

        parquet_files = sorted(list((self.data_root / "data" / "chunk-000").glob("*.parquet")))[:max_files]
        print(f"Loading {len(parquet_files)} demonstration parquet files for DINOv2 training...")

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

                for i in range(0, N - horizon, 2):
                    img_raw = ep_df["observation.images.image"].iloc[i]
                    state = ep_df["observation.state"].iloc[i]
                    actions = np.stack([ep_df["action"].iloc[i + h] for h in range(horizon)])  # [8, 7]

                    self.samples.append({
                        "img_bytes": img_raw["bytes"],
                        "task_prompt": task_str,
                        "eef_pos": state[:3].astype(np.float32),
                        "gripper_q": state[7:9].astype(np.float32) if len(state) > 8 else state[-2:].astype(np.float32),
                        "actions": actions.astype(np.float32)
                    })

        print(f"Loaded {len(self.samples)} trajectory chunks for DINOv2-ACT training!")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]

        # Decode image
        pil_img = Image.open(io.BytesIO(item["img_bytes"])).convert("RGB").resize((128, 128))
        if self.augment and random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(random.uniform(0.9, 1.1))

        img_np = np.array(pil_img, dtype=np.float32) / 255.0  # [128, 128, 3]
        img_tensor = torch.tensor(img_np).permute(2, 0, 1)  # [3, 128, 128]

        return {
            "rgb": img_tensor,
            "prompt": item["task_prompt"],
            "eef_pos": torch.tensor(item["eef_pos"], dtype=torch.float32),
            "gripper_q": torch.tensor(item["gripper_q"], dtype=torch.float32),
            "actions": torch.tensor(item["actions"], dtype=torch.float32)  # [8, 7]
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "eef_pos": torch.stack([b["eef_pos"] for b in batch]),
        "gripper_q": torch.stack([b["gripper_q"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch])
    }


def train_dinov2_act(
    data_root: str = "/media/kavinder/hdd2/datasets/libero/libero_spatial",
    output_dir: str = "/media/kavinder/hdd2/geo_jepa_runs/dinov2_act_policy",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cuda"
):
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" GEO-JEPA: TRAINING DINOv2-GROUNDED MULTIMODAL ACT POLICY")
    print(f" Output Directory: {output_dir}")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 85)

    dataset = DINOv2LiberoDataset(data_root=data_root, horizon=8, max_files=40, augment=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    policy = DINOv2ACTPolicy(embed_dim=384, action_dim=7, horizon=8).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader), eta_min=1e-5)

    policy.train()
    global_step = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{epochs:02d}")

        for batch in pbar:
            global_step += 1
            rgb = batch["rgb"].to(device)
            prompts = batch["prompts"]
            eef_pos = batch["eef_pos"].to(device)
            gripper_q = batch["gripper_q"].to(device)
            gt_actions = batch["actions"].to(device)  # [B, 8, 7]

            optimizer.zero_grad()

            # Normalize target actions
            gt_norm_actions = policy.normalize_actions(gt_actions)

            # Forward pass
            pred_norm_actions = policy(
                rgb_image=rgb,
                task_prompts=prompts,
                eef_pos=eef_pos,
                gripper_q=gripper_q
            )

            # L1 + L2 Action Chunk Loss
            loss_l1 = F.l1_loss(pred_norm_actions, gt_norm_actions)
            loss_l2 = F.mse_loss(pred_norm_actions, gt_norm_actions)
            loss = loss_l1 + 0.5 * loss_l2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "l1": f"{loss_l1.item():.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Complete | Mean ACT Loss: {avg_loss:.5f} | Elapsed: {elapsed:.1f}s")

        if epoch % 5 == 0 or epoch == epochs:
            ckpt_path = ckpt_dir / f"dinov2_act_epoch_{epoch:03d}.pt"
            latest_path = ckpt_dir / "dinov2_act_latest.pt"
            save_dict = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }
            torch.save(save_dict, ckpt_path)
            torch.save(save_dict, latest_path)
            print(f"--> [CHECKPOINT SAVED] {ckpt_path}")

    print("\n" + "=" * 85)
    print(f" DINOv2 ACT TRAINING COMPLETE in {time.time()-start_time:.1f}s!")
    print(f" Final Checkpoint: {ckpt_dir / 'dinov2_act_latest.pt'}")
    print("=" * 85)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dinov2_act(epochs=25, batch_size=32, lr=3e-4, device=device)
