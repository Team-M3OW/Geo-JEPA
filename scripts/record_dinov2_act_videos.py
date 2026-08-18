#!/usr/bin/env python3
"""
Geo-JEPA: Record High-Definition Rollout Videos for Trained DINOv2-ACT Policy & Sync to Remote.

Records rollout videos showing successful goal completion:
- Task 1: 90% Success (Bowl between plate & ramekin)
- Task 2: 90% Success (Bowl next to ramekin)
- Task 4: 90% Success (Bowl on cookie box obstacle)
- Task 5: 100% Success (Bowl in top drawer cavity)
- Task 7: 90% Success (Bowl next to cookie box)
- Task 9: 60% Success (Bowl next to plate)

Destination: cha0s@10.141.90.48:~/geo_jepa_eval_results/dinov2_act_videos/
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import paramiko
from scp import SCPClient
import torch

sys.path.insert(0, "/home/kavinder/LIBERO")
sys.path.insert(0, "/home/kavinder/Geo-JEPA")

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from geo_jepa.models.dinov2_act_vla_policy import DINOv2ACTPolicy


def draw_act_hud(img: np.ndarray, prompt: str, step: int, is_success: bool) -> np.ndarray:
    canvas = img.copy()
    canvas = cv2.resize(canvas, (400, 400))
    h, w, _ = canvas.shape

    # Header banner
    header_col = (40, 160, 40) if is_success else (160, 80, 40)
    cv2.rectangle(canvas, (0, 0), (w, 36), header_col, -1)
    cv2.putText(canvas, "DINOv2-ACT (Pretrained Multimodal VLA)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Footer overlay
    cv2.rectangle(canvas, (0, h - 60), (w, h), (0, 0, 0), -1)
    succ_txt = "BDDL GOAL: SUCCESS [SATISFIED]" if is_success else "BDDL GOAL: IN PROGRESS"
    succ_col = (0, 255, 0) if is_success else (200, 200, 200)

    cv2.putText(canvas, succ_txt, (10, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, succ_col, 1)
    cv2.putText(canvas, f"Step: {step:03d} / 150", (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    return canvas


def record_dinov2_videos():
    out_dir = Path("/media/kavinder/hdd2/geo_jepa_eval_results/dinov2_act_videos")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = DINOv2ACTPolicy(embed_dim=384, action_dim=7, horizon=8).to(device)
    ckpt_path = "/media/kavinder/hdd2/geo_jepa_runs/dinov2_act_policy/checkpoints/dinov2_act_latest.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    benchmark = get_benchmark("libero_spatial")()
    num_tasks = benchmark.get_num_tasks()

    print("=" * 85)
    print(" GEO-JEPA: RENDERING DINOv2-ACT BENCHMARK ROLLOUT VIDEOS")
    print(f" Output Directory: {out_dir}")
    print("=" * 85)

    for idx in range(num_tasks):
        task = benchmark.get_task(idx)
        clean_name = f"dinov2_act_task_{idx+1:02d}_{task.name[:35]}.mp4"
        out_video = str(out_dir / clean_name)

        print(f"\n[{idx+1}/{num_tasks}] Rendering Video: {clean_name}")
        print(f"  Prompt: \"{task.language}\"")

        bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
        init_states = torch.load(init_path, weights_only=False)

        env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)
        env.set_init_state(init_states[0])
        for _ in range(5): obs, _, _, _ = env.step(np.zeros(7))

        frames = []
        is_success = False

        for step in range(145):
            raw_rgb = obs["agentview_image"][::-1, :, :]
            img_t = torch.tensor(raw_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            eef_t = torch.tensor(obs["robot0_eef_pos"], dtype=torch.float32, device=device).unsqueeze(0)
            grp_t = torch.tensor(obs["robot0_gripper_qpos"], dtype=torch.float32, device=device).unsqueeze(0)

            act_chunk = policy.get_action_chunk(
                rgb_image=img_t,
                task_prompt=task.language,
                eef_pos=eef_t,
                gripper_q=grp_t
            )

            # Step 1 sub-step of chunk
            obs, r, done, info = env.step(np.clip(act_chunk[0], -1.0, 1.0))
            if env.check_success():
                is_success = True

            hud_frame = draw_act_hud(raw_rgb, task.language, step, is_success)
            frames.append(hud_frame)

        env.close()

        # Save MP4
        writer = imageio.get_writer(out_video, fps=20, quality=8)
        for f in frames: writer.append_data(f)
        writer.close()
        print(f"  --> Saved Video: {clean_name} (Success: {is_success})")

    # Sync to remote server
    print("\n" + "=" * 85)
    print(" SYNCHRONIZING DINOv2-ACT VIDEOS TO REMOTE SERVER (cha0s@10.141.90.48)...")
    print("=" * 85)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect("10.141.90.48", username="cha0s", password="arshabbas06", timeout=30)
        ssh.exec_command("mkdir -p ~/geo_jepa_eval_results/dinov2_act_videos")
        with SCPClient(ssh.get_transport(), socket_timeout=120.0) as scp:
            for f in sorted(os.listdir(out_dir)):
                scp.put(str(out_dir / f), remote_path="~/geo_jepa_eval_results/dinov2_act_videos/")
        stdin, stdout, stderr = ssh.exec_command("ls -lh ~/geo_jepa_eval_results/dinov2_act_videos/")
        print(stdout.read().decode())
        ssh.close()
        print("All DINOv2-ACT videos successfully synced to remote server!")
    except Exception as e:
        print(f"Remote sync warning: {e}")


if __name__ == "__main__":
    record_dinov2_videos()
