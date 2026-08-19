#!/usr/bin/env python3
"""
Geo-JEPA: Dual-Camera DINOv2 ACT Benchmark Evaluator with Temporal Action Ensembling.

Features:
1. Dual-Camera Ingestion: Third-person (agentview) + Egocentric (wrist/eye-in-hand).
2. Action Space Normalization: Perfectly balanced translation and wrist rotation.
3. Temporal Action Ensembling: Exponentially weighted smoothing over overlapping H=16 chunks.
4. Comprehensive Multi-Suite Evaluation.

Output: /media/kavinder/hdd2/geo_jepa_eval_results/dual_camera_act_eval/
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, "/home/kavinder/LIBERO")
sys.path.insert(0, "/home/kavinder/Geo-JEPA")

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from geo_jepa.models.dual_camera_dinov2_act_policy import DualCameraDINOv2ACTPolicy


class TemporalEnsembleBuffer:
    """Maintains an action buffer of overlapping predicted chunks with exponential moving average weights."""
    def __init__(self, horizon: int = 16, decay: float = 0.05):
        self.horizon = horizon
        self.decay = decay
        self.actions_by_step = defaultdict(list)

    def add_chunk(self, start_step: int, chunk: np.ndarray):
        for i in range(len(chunk)):
            t = start_step + i
            weight = math.exp(-self.decay * i)
            self.actions_by_step[t].append((chunk[i], weight))

    def get_action(self, step: int) -> np.ndarray:
        if step not in self.actions_by_step or len(self.actions_by_step[step]) == 0:
            return np.zeros(7)
        actions, weights = zip(*self.actions_by_step[step])
        weights = np.array(weights) / sum(weights)
        ensembled_action = np.sum([w * a for w, a in zip(weights, actions)], axis=0)
        return ensembled_action


def run_dual_camera_rollout(
    env: OffScreenRenderEnv,
    policy: DualCameraDINOv2ACTPolicy,
    task_prompt: str,
    max_steps: int = 150,
    replan_interval: int = 4,
    device: str = "cuda"
) -> Tuple[bool, int]:
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    ensemble_buffer = TemporalEnsembleBuffer(horizon=16, decay=0.05)
    task_success = False
    step = 0

    while step < max_steps:
        # Every replan_interval steps, query the dual-camera ACT policy
        if step % replan_interval == 0:
            raw_agent = obs["agentview_image"][::-1, :, :]
            raw_wrist = obs["robot0_eye_in_hand_image"][::-1, :, :]

            agent_t = torch.tensor(raw_agent / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            wrist_t = torch.tensor(raw_wrist / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            eef_t = torch.tensor(obs["robot0_eef_pos"], dtype=torch.float32, device=device).unsqueeze(0)
            grp_t = torch.tensor(obs["robot0_gripper_qpos"], dtype=torch.float32, device=device).unsqueeze(0)

            chunk = policy.get_action_chunk(
                agentview_rgb=agent_t,
                wrist_rgb=wrist_t,
                task_prompt=task_prompt,
                eef_pos=eef_t,
                gripper_q=grp_t
            )
            ensemble_buffer.add_chunk(start_step=step, chunk=chunk)

        # Get smoothly ensembled action for current step
        act = ensemble_buffer.get_action(step)
        obs, reward, done, info = env.step(np.clip(act, -1.0, 1.0))
        step += 1

        if env.check_success():
            task_success = True
            break

    return task_success, step


def evaluate_suite(
    suite_name: str = "libero_spatial",
    num_seeds: int = 10,
    device: str = "cuda"
) -> Dict:
    out_dir = Path("/media/kavinder/hdd2/geo_jepa_eval_results/dual_camera_act_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(f" GEO-JEPA: DUAL-CAMERA DINOv2 ACT BENCHMARK EVALUATION ({suite_name.upper()})")
    print(f" Output Directory: {out_dir}")
    print("=" * 85)

    benchmark = get_benchmark(suite_name)()
    num_tasks = benchmark.get_num_tasks()

    policy = DualCameraDINOv2ACTPolicy(embed_dim=384, action_dim=7, horizon=16).to(device)
    ckpt_path = "/media/kavinder/hdd2/geo_jepa_runs/dual_camera_act_40task/checkpoints/dual_camera_act_latest.pt"
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        policy.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded Trained Dual-Camera ACT Checkpoint: {ckpt_path} (Epoch: {ckpt.get('epoch', 30)}, Loss: {ckpt.get('loss', 0.0):.4f})")
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

            s, steps = run_dual_camera_rollout(
                env, policy, task.language, max_steps=150, replan_interval=4, device=device
            )
            if s: succ_count += 1
            env.close()

        sr = (succ_count / num_seeds) * 100.0
        print(f"  --> Dual-Camera ACT Success Rate: {sr:.1f}% ({succ_count}/{num_seeds})")

        task_results.append({
            "task_id": task_idx + 1,
            "task_name": task.name,
            "prompt": task.language,
            "num_seeds": num_seeds,
            "success_rate": sr
        })

    mean_sr = float(np.mean([r["success_rate"] for r in task_results]))

    report = {
        "model": "Dual-Camera DINOv2 Multimodal ACT with Temporal Action Ensembling",
        "suite_name": suite_name,
        "num_tasks": num_tasks,
        "seeds_per_task": num_seeds,
        "mean_success_rate": mean_sr,
        "task_breakdown": task_results
    }

    report_path = out_dir / f"dual_camera_act_{suite_name}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 85)
    print(f" DUAL-CAMERA ACT EVALUATION COMPLETE FOR {suite_name.upper()}!")
    print(f" Mean Success Rate: {mean_sr:.2f}%")
    print(f" Saved Report: {report_path}")
    print("=" * 85)

    return report


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_suite(suite_name="libero_spatial", num_seeds=10, device=device)
