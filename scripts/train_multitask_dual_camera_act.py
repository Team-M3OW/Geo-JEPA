#!/usr/bin/env python3
"""
Geo-JEPA: Unified 40-Task Dual-Camera DINOv2 ACT Training Pipeline.

Features:
1. Multi-Task Training Across All 4 LIBERO Benchmark Suites (40 Tasks).
2. Dual-Camera Ingestion: Third-person (AgentView) + Egocentric (Wrist/Eye-in-Hand).
3. Pretrained DINOv2 Visual Tokenizer.
4. Action Space Normalization & Horizon H=16 Action Chunking.
5. Composite Loss: L1/L2 Loss on Arm Velocities + Binary Cross-Entropy on Gripper.

Output: /media/kavinder/hdd2/geo_jepa_runs/dual_camera_act_40task/checkpoints/
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

from geo_jepa.models.dual_camera_dinov2_act_policy import DualCameraDINOv2ACTPolicy


class MultiTaskDualCameraDataset(Dataset):
    def __init__(self, libero_root: str, horizon: int = 16, max_files_per_suite: int = 25, augment: bool = True):
        self.libero_root = Path(libero_root)
        self.horizon = horizon
        self.augment = augment
        self.samples = []

        suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
        print(f"Loading Multi-Task Dual-Camera Data across suites: {suites}...")

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

                    for i in range(0, N - horizon, 3):  # Stride 3 for balanced dataset
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

        print(f"Total Multi-Task Dual-Camera Trajectory Chunks Loaded: {len(self.samples)}!")

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

        # Decode AgentView Image
        pil_agent = Image.open(io.BytesIO(item["agent_bytes"])).convert("RGB").resize((128, 128))
        pil_agent = self._augment(pil_agent)
        agent_tensor = torch.tensor(np.array(pil_agent, dtype=np.float32) / 255.0).permute(2, 0, 1)

        # Decode Wrist Image
        pil_wrist = Image.open(io.BytesIO(item["wrist_bytes"])).convert("RGB").resize((128, 128))
        pil_wrist = self._augment(pil_wrist)
        wrist_tensor = torch.tensor(np.array(pil_wrist, dtype=np.float32) / 255.0).permute(2, 0, 1)

        return {
            "agent_rgb": agent_tensor,
            "wrist_rgb": wrist_tensor,
            "prompt": item["prompt"],
            "eef_pos": torch.tensor(item["eef_pos"], dtype=torch.float32),
            "gripper_q": torch.tensor(item["gripper_q"], dtype=torch.float32),
            "actions": torch.tensor(item["actions"], dtype=torch.float32)  # [16, 7]
        }


def collate_dual_camera(batch: List[Dict]) -> Dict:
    return {
        "agent_rgb": torch.stack([b["agent_rgb"] for b in batch]),
        "wrist_rgb": torch.stack([b["wrist_rgb"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "eef_pos": torch.stack([b["eef_pos"] for b in batch]),
        "gripper_q": torch.stack([b["gripper_q"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch])
    }


def train_dual_camera_act(
    libero_root: str = "/media/kavinder/hdd2/datasets/libero",
    output_dir: str = "/media/kavinder/hdd2/geo_jepa_runs/dual_camera_act_40task",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cuda"
):
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" GEO-JEPA: UNIFIED 40-TASK DUAL-CAMERA DINOv2 ACT TRAINING")
    print(f" Output Directory: {output_dir}")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 85)

    dataset = MultiTaskDualCameraDataset(libero_root=libero_root, horizon=16, max_files_per_suite=25, augment=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_dual_camera,
        num_workers=4,
        pin_memory=True
    )

    policy = DualCameraDINOv2ACTPolicy(embed_dim=384, action_dim=7, horizon=16).to(device)
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
            gt_actions = batch["actions"].to(device)  # [B, 16, 7]

            optimizer.zero_grad()

            # Normalize target arm actions
            gt_arm_norm, gt_gripper = policy.normalize_actions(gt_actions)

            # Forward pass
            pred_arm_norm, pred_gripper = policy(
                agentview_rgb=agent_rgb,
                wrist_rgb=wrist_rgb,
                task_prompts=prompts,
                eef_pos=eef_pos,
                gripper_q=gripper_q
            )

            # Arm loss (L1 + L2)
            loss_arm = F.l1_loss(pred_arm_norm, gt_arm_norm) + 0.5 * F.mse_loss(pred_arm_norm, gt_arm_norm)
            # Gripper loss
            loss_gripper = F.mse_loss(pred_gripper, gt_gripper)
            loss = loss_arm + 0.5 * loss_gripper

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "arm": f"{loss_arm.item():.4f}", "grp": f"{loss_gripper.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Complete | Mean Dual-Camera ACT Loss: {avg_loss:.5f} | Elapsed: {elapsed:.1f}s")

        if epoch % 5 == 0 or epoch == epochs:
            ckpt_path = ckpt_dir / f"dual_camera_act_epoch_{epoch:03d}.pt"
            latest_path = ckpt_dir / "dual_camera_act_latest.pt"
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
    print(f" DUAL-CAMERA 40-TASK ACT TRAINING COMPLETE in {time.time()-start_time:.1f}s!")
    print(f" Final Checkpoint: {ckpt_dir / 'dual_camera_act_latest.pt'}")
    print("=" * 85)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dual_camera_act(epochs=30, batch_size=32, lr=3e-4, device=device)
