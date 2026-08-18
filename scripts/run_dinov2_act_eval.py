#!/usr/bin/env python3
"""
Geo-JEPA: DINOv2-Grounded Multimodal ACT Policy Receding Horizon Evaluator.

Evaluates:
- Pretrained DINOv2 Vision Backbone
- Action Normalization / Denormalization
- Direct Action Chunking Transformer with Receding Horizon MPC (H=8, K=4)

Output: /media/kavinder/hdd2/geo_jepa_eval_results/dinov2_act_eval/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, "/home/kavinder/LIBERO")
sys.path.insert(0, "/home/kavinder/Geo-JEPA")

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from geo_jepa.models.dinov2_act_vla_policy import DINOv2ACTPolicy


def run_act_rollout(
    env: OffScreenRenderEnv,
    policy: DINOv2ACTPolicy,
    task_prompt: str,
    max_steps: int = 150,
    chunk_exec_steps: int = 4,
    device: str = "cuda"
) -> Tuple[bool, int]:
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    task_success = False
    step = 0

    while step < max_steps:
        # Preprocess camera frame & proprioception
        raw_rgb = obs["agentview_image"][::-1, :, :]  # [128, 128, 3] uint8
        img_tensor = torch.tensor(raw_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        eef_pos = torch.tensor(obs["robot0_eef_pos"], dtype=torch.float32, device=device).unsqueeze(0)
        gripper_q = torch.tensor(obs["robot0_gripper_qpos"], dtype=torch.float32, device=device).unsqueeze(0)

        # Get denormalized action chunk
        act_chunk = policy.get_action_chunk(
            rgb_image=img_tensor,
            task_prompt=task_prompt,
            eef_pos=eef_pos,
            gripper_q=gripper_q
        )  # [8, 7]

        # Execute first K=4 steps of chunk
        for k in range(min(chunk_exec_steps, len(act_chunk))):
            obs, reward, done, info = env.step(np.clip(act_chunk[k], -1.0, 1.0))
            step += 1

            if env.check_success():
                task_success = True
                break

        if task_success or step >= max_steps:
            break

    return task_success, step


def evaluate_dinov2_act(
    suite_name: str = "libero_spatial",
    num_seeds: int = 10,
    device: str = "cuda"
) -> Dict:
    output_dir = Path("/media/kavinder/hdd2/geo_jepa_eval_results/dinov2_act_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" GEO-JEPA: DINOv2 MULTIMODAL ACT BENCHMARK EVALUATION")
    print(f" Suite: {suite_name} | Seeds per Task: {num_seeds}")
    print(f" Output Directory: {output_dir}")
    print("=" * 85)

    benchmark = get_benchmark(suite_name)()
    num_tasks = benchmark.get_num_tasks()

    policy = DINOv2ACTPolicy(embed_dim=384, action_dim=7, horizon=8).to(device)
    ckpt_path = "/media/kavinder/hdd2/geo_jepa_runs/dinov2_act_policy/checkpoints/dinov2_act_latest.pt"
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        policy.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded Trained DINOv2-ACT Checkpoint: {ckpt_path} (Epoch: {ckpt.get('epoch', 25)}, Loss: {ckpt.get('loss', 0.0):.4f})")
    policy.eval()

    task_results = []

    for task_idx in range(num_tasks):
        task = benchmark.get_task(task_idx)
        bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
        init_states = torch.load(init_path, weights_only=False)

        print(f"\n[{task_idx+1}/{num_tasks}] [{suite_name}] Task: {task.name}")
        print(f"  Language Instruction: \"{task.language}\"")

        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
        }

        succ_count = 0

        for seed in range(num_seeds):
            init_state = init_states[seed % len(init_states)]
            env = OffScreenRenderEnv(**env_args)
            env.set_init_state(init_state)
            
            s, steps = run_act_rollout(
                env, policy, task.language, chunk_exec_steps=4, device=device
            )
            if s: succ_count += 1
            env.close()

        sr = (succ_count / num_seeds) * 100.0
        print(f"  --> DINOv2-ACT Success Rate: {sr:.1f}% ({succ_count}/{num_seeds})")

        task_results.append({
            "task_id": task_idx + 1,
            "task_name": task.name,
            "prompt": task.language,
            "num_seeds": num_seeds,
            "success_rate": sr
        })

    mean_sr = float(np.mean([r["success_rate"] for r in task_results]))

    report = {
        "model": "DINOv2 Multimodal Action Chunking Transformer (ACT)",
        "suite_name": suite_name,
        "num_tasks": num_tasks,
        "seeds_per_task": num_seeds,
        "mean_success_rate": mean_sr,
        "task_breakdown": task_results
    }

    report_path = output_dir / "dinov2_act_libero_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 85)
    print(" DINOv2-ACT EVALUATION COMPLETE!")
    print(f" Mean Success Rate: {mean_sr:.2f}%")
    print(f" Saved Report: {report_path}")
    print("=" * 85)

    return report


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_dinov2_act(suite_name="libero_spatial", num_seeds=10, device=device)
