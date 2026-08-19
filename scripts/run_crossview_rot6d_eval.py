#!/usr/bin/env python3
"""
Geo-JEPA: Comprehensive 4-Suite Benchmark Evaluator for Cross-View Attention + 6D Continuous Rotation ACT Policy.

Evaluates across:
1. LIBERO-Spatial (10 Tasks)
2. LIBERO-Object  (10 Tasks)
3. LIBERO-Goal    (10 Tasks)
4. LIBERO-10      (10 Tasks)

Features:
- Fix 3: Cross-View Attention Bridge (AgentView ◄► Wrist) with 2D Spatial Grid Coordinates.
- Fix 4: 6D Continuous Rotation Head (Gram-Schmidt Orthogonalization to SO(3)).
- Temporal Action Ensembling (H=16, K=4).

Output: /media/kavinder/hdd2/geo_jepa_eval_results/crossview_rot6d_eval/
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
import paramiko
from scp import SCPClient
import torch

sys.path.insert(0, "/home/kavinder/LIBERO")
sys.path.insert(0, "/home/kavinder/Geo-JEPA")

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from geo_jepa.models.crossview_rot6d_act_policy import CrossViewRot6dACTPolicy


class TemporalEnsembleBuffer:
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
        return np.sum([w * a for w, a in zip(weights, actions)], axis=0)


def run_crossview_rot6d_rollout(
    env: OffScreenRenderEnv,
    policy: CrossViewRot6dACTPolicy,
    task_prompt: str,
    max_steps: int = 160,
    replan_interval: int = 4,
    device: str = "cuda"
) -> Tuple[bool, int]:
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    ensemble_buffer = TemporalEnsembleBuffer(horizon=16, decay=0.05)
    task_success = False
    step = 0

    while step < max_steps:
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

        act = ensemble_buffer.get_action(step)
        obs, reward, done, info = env.step(np.clip(act, -1.0, 1.0))
        step += 1

        if env.check_success():
            task_success = True
            break

    return task_success, step


def evaluate_all_suites(num_seeds: int = 10, device: str = "cuda"):
    out_dir = Path("/media/kavinder/hdd2/geo_jepa_eval_results/crossview_rot6d_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" GEO-JEPA: COMPREHENSIVE 4-SUITE CROSS-VIEW 6D ROTATION ACT EVALUATION")
    print(f" Output Directory: {out_dir}")
    print("=" * 85)

    policy = CrossViewRot6dACTPolicy(embed_dim=384, horizon=16).to(device)
    ckpt_path = "/media/kavinder/hdd2/geo_jepa_runs/crossview_rot6d_act/checkpoints/crossview_rot6d_latest.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded Trained Checkpoint: {ckpt_path} (Epoch: {ckpt.get('epoch', 30)}, Loss: {ckpt.get('loss', 0.0):.4f})")
    policy.eval()

    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    all_suite_results = {}

    for suite_name in suites:
        print("\n" + "#" * 85)
        print(f" EVALUATING SUITE: {suite_name.upper()}")
        print("#" * 85)

        benchmark = get_benchmark(suite_name)()
        num_tasks = benchmark.get_num_tasks()
        task_results = []

        for task_idx in range(num_tasks):
            task = benchmark.get_task(task_idx)
            bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            init_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
            init_states = torch.load(init_path, weights_only=False)

            print(f"\n[{task_idx+1}/{num_tasks}] [{suite_name}] Task: {task.name}")
            print(f"  Prompt: \"{task.language}\"")

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

                s, steps = run_crossview_rot6d_rollout(
                    env, policy, task.language, max_steps=160, replan_interval=4, device=device
                )
                if s: succ_count += 1
                env.close()

            sr = (succ_count / num_seeds) * 100.0
            print(f"  --> Success Rate: {sr:.1f}% ({succ_count}/{num_seeds})")

            task_results.append({
                "task_id": task_idx + 1,
                "task_name": task.name,
                "prompt": task.language,
                "num_seeds": num_seeds,
                "success_rate": sr
            })

        mean_sr = float(np.mean([r["success_rate"] for r in task_results]))
        all_suite_results[suite_name] = {
            "mean_success_rate": mean_sr,
            "tasks": task_results
        }
        print(f"\n>>> [{suite_name.upper()}] SUITE MEAN SUCCESS RATE: {mean_sr:.2f}% <<<")

    overall_mean = float(np.mean([v["mean_success_rate"] for v in all_suite_results.values()]))
    comprehensive_report = {
        "model": "Cross-View Attention Bridge + 6D Continuous Rotation ACT Policy",
        "checkpoint": ckpt_path,
        "overall_mean_success_rate": overall_mean,
        "suites": all_suite_results
    }

    report_path = out_dir / "comprehensive_crossview_rot6d_report.json"
    with open(report_path, "w") as f:
        json.dump(comprehensive_report, f, indent=2)

    print("\n" + "=" * 85)
    print(" ALL 4 SUITES EVALUATION COMPLETE!")
    print(f" OVERALL 40-TASK BENCHMARK SUCCESS RATE: {overall_mean:.2f}%")
    print(f" Saved Report: {report_path}")
    print("=" * 85)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_all_suites(num_seeds=10, device=device)
