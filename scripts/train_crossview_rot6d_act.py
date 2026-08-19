#!/usr/bin/env python3
"""
Geo-JEPA: Training Pipeline for Cross-View Attention Bridge + 6D Continuous Rotation ACT Policy.

Implements:
1. Spatial 2D Coordinate Grid Encodings.
2. Bidirectional Cross-View Attention (AgentView ◄► Wrist).
3. 6D Continuous Rotation Target Conversion & SO(3) Loss.
4. Translation Space Normalization.

Dataset: /media/kavinder/hdd2/datasets/libero/
Checkpoints: /media/kavinder/hdd2/geo_jepa_runs/crossview_rot6d_act/checkpoints/
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

from geo_jepa.models.crossview_rot6d_act_policy import (
    CrossViewRot6dACTPolicy,
    euler_to_rot6d,
    compute_rotation_matrix_from_ortho6d
)


class MultiSuiteRot6dDataset(Dataset):
    def __init__(self, libero_root: str, horizon: int = 16, max_files_per_suite: int = 30, augment: bool = True):
        self.libero_root = Path(libero_root)
        self.horizon = horizon
        self.augment = augment
        self.samples = []

        suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
        print(f"Loading Cross-View 6D Rotation Data across: {suites}...")

        for s in suites:
            s_dir = self.libero_root / s
            if not s_dir.exists():
                continue
            tasks_df = pd.read_parquet(s_dir / "meta" / "tasks.parquet")
            task_dict = {row["task_index"]: t_name for t_name, row in tasks_df.iterrows()}

            parquet_files = sorted(list((s_dir / "data" / "chunk-000").glob("*.parquet")))[:max_files_per_suite]

            for pf in parquet_files:
                df = pd.read_parquet(pf)
                for ep in df["episode_index"].unique():
                    ep_df = df[df["episode_index"] == ep].reset_index(drop=True)
                    N = len(ep_df)
                    if N <= horizon:
                        continue
                    t_idx = int(ep_df["task_index"].iloc[0])
                    prompt_str = task_dict.get(t_idx, "manipulate the target object")

                    for i in range(0, N - horizon, 2):
                        agent_raw = ep_df["observation.images.image"].iloc[i]
                        wrist_raw = ep_df["observation.images.wrist_image"].iloc[i]
                        state = ep_df["observation.state"].iloc[i]
                        actions = np.stack([ep_df["action"].iloc[i + h] for h in range(horizon)])  # [16, 7]

                        self.samples.append({
                            "agent_bytes": agent_raw["bytes"],
                            "wrist_bytes": wrist_raw["bytes"],
                            "prompt": prompt_str,
                            "eef_pos": state[:3].astype(np.float32),
                            "gripper_q": state[7:9].astype(np.float32) if len(state) > 8 else state[-2:].astype(np.float32),
                            "actions": actions.astype(np.float32)
                        })

        print(f"Total Cross-View Trajectory Samples Loaded: {len(self.samples)}!")

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, pil_img: Image.Image) -> Image.Image:
        if not self.augment: return pil_img
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(random.uniform(0.9, 1.1))
        return pil_img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]

        # Decode AgentView
        pil_agent = Image.open(io.BytesIO(item["agent_bytes"])).convert("RGB").resize((128, 128))
        pil_agent = self._augment(pil_agent)
        agent_tensor = torch.tensor(np.array(pil_agent, dtype=np.float32) / 255.0).permute(2, 0, 1)

        # Decode Wrist
        pil_wrist = Image.open(io.BytesIO(item["wrist_bytes"])).convert("RGB").resize((128, 128))
        pil_wrist = self._augment(pil_wrist)
        wrist_tensor = torch.tensor(np.array(pil_wrist, dtype=np.float32) / 255.0).permute(2, 0, 1)

        raw_actions = torch.tensor(item["actions"], dtype=torch.float32)  # [16, 7]
        gt_trans = raw_actions[:, :3]                                     # [16, 3]
        gt_euler = raw_actions[:, 3:6]                                    # [16, 3]
        gt_rot6d = euler_to_rot6d(gt_euler)                               # [16, 6]
        gt_gripper = raw_actions[:, 6:7]                                  # [16, 1]

        return {
            "agent_rgb": agent_tensor,
            "wrist_rgb": wrist_tensor,
            "prompt": item["prompt"],
            "eef_pos": torch.tensor(item["eef_pos"], dtype=torch.float32),
            "gripper_q": torch.tensor(item["gripper_q"], dtype=torch.float32),
            "gt_trans": gt_trans,
            "gt_rot6d": gt_rot6d,
            "gt_gripper": gt_gripper
        }


def collate_rot6d(batch: List[Dict]) -> Dict:
    return {
        "agent_rgb": torch.stack([b["agent_rgb"] for b in batch]),
        "wrist_rgb": torch.stack([b["wrist_rgb"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "eef_pos": torch.stack([b["eef_pos"] for b in batch]),
        "gripper_q": torch.stack([b["gripper_q"] for b in batch]),
        "gt_trans": torch.stack([b["gt_trans"] for b in batch]),
        "gt_rot6d": torch.stack([b["gt_rot6d"] for b in batch]),
        "gt_gripper": torch.stack([b["gt_gripper"] for b in batch])
    }


def train_crossview_rot6d(
    libero_root: str = "/media/kavinder/hdd2/datasets/libero",
    output_dir: str = "/media/kavinder/hdd2/geo_jepa_runs/crossview_rot6d_act",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cuda"
):
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" GEO-JEPA: TRAINING CROSS-VIEW ATTENTION + 6D ROTATION ACT POLICY")
    print(f" Output Directory: {output_dir}")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 85)

    dataset = MultiSuiteRot6dDataset(libero_root=libero_root, horizon=16, max_files_per_suite=30, augment=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_rot6d,
        num_workers=4,
        pin_memory=True
    )

    policy = CrossViewRot6dACTPolicy(embed_dim=384, horizon=16).to(device)
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
            agent_rgb = batch["agent_rgb"].to(device)
            wrist_rgb = batch["wrist_rgb"].to(device)
            prompts = batch["prompts"]
            eef_pos = batch["eef_pos"].to(device)
            gripper_q = batch["gripper_q"].to(device)

            gt_trans = batch["gt_trans"].to(device)
            gt_rot6d = batch["gt_rot6d"].to(device)
            gt_gripper = batch["gt_gripper"].to(device)

            optimizer.zero_grad()

            # Normalize translation target
            gt_trans_norm = (gt_trans - policy.trans_mean) / (policy.trans_std + 1e-4)

            # Forward pass
            preds = policy(
                agentview_rgb=agent_rgb,
                wrist_rgb=wrist_rgb,
                task_prompts=prompts,
                eef_pos=eef_pos,
                gripper_q=gripper_q
            )

            # 1. Translation Loss (L1 + 0.5 * L2)
            loss_trans = F.l1_loss(preds["pred_trans_norm"], gt_trans_norm) + 0.5 * F.mse_loss(preds["pred_trans_norm"], gt_trans_norm)

            # 2. 6D Continuous Rotation Loss (Smooth SO(3) Orthogonal Distance)
            loss_rot = F.mse_loss(preds["pred_rot6d"], gt_rot6d)

            # 3. Gripper BCE / Regression Loss
            loss_gripper = F.mse_loss(preds["pred_gripper"], gt_gripper)

            # Total Composite Loss
            loss = loss_trans + 1.0 * loss_rot + 0.5 * loss_gripper

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "trans": f"{loss_trans.item():.4f}", "rot6d": f"{loss_rot.item():.4f}", "grp": f"{loss_gripper.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Complete | Mean Loss: {avg_loss:.5f} | Elapsed: {elapsed:.1f}s")

        if epoch % 5 == 0 or epoch == epochs:
            ckpt_path = ckpt_dir / f"crossview_rot6d_epoch_{epoch:03d}.pt"
            latest_path = ckpt_dir / "crossview_rot6d_latest.pt"
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
    print(f" CROSS-VIEW 6D ROTATION ACT TRAINING COMPLETE in {time.time()-start_time:.1f}s!")
    print(f" Final Checkpoint: {ckpt_dir / 'crossview_rot6d_latest.pt'}")
    print("=" * 85)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_crossview_rot6d(epochs=30, batch_size=32, lr=3e-4, device=device)
