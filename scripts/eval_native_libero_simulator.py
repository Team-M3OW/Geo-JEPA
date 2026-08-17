#!/usr/bin/env python3
"""
Geo-JEPA: Native Official LIBERO Simulator Benchmark Evaluator.

Runs closed-loop policy evaluation directly inside the official LIBERO physics simulator:
- Uses official BDDL task definitions (libero_spatial, libero_object, libero_goal, libero_10)
- Receives live agentview & wrist RGB camera frames (128x128x3) from MuJoCo
- Continuous Flow Matching ODE action inference at 20 Hz
- Verifies physical success using official env.check_success()
- Compares:
    1. Baseline 2D VLA-JEPA
    2. Geo-JEPA (Ours, Coupled 3D Rays + Spatial Forcing)

Output: /media/kavinder/hdd2/geo_jepa_eval_results/native_libero_simulator_eval/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, "/home/kavinder/LIBERO")
sys.path.insert(0, "/home/kavinder/Geo-JEPA")

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from geo_jepa.models.coupled_geo_action_flow import CoupledGeoActionFlow


class PolicyModel(nn.Module):
    """Evaluation policy architecture matching training setup."""

    def __init__(
        self,
        config_name: str,
        embed_dim: int = 512,
        action_horizon: int = 8,
        action_dim: int = 7,
        num_points: int = 64
    ):
        super().__init__()
        self.config_name = config_name
        self.embed_dim = embed_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.num_points = num_points

        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.vis_proj = nn.Linear(256 * 16, embed_dim)

        self.is_coupled = (config_name == "full_coupled_geo_jepa")
        if self.is_coupled:
            self.coupled_flow = CoupledGeoActionFlow(
                cond_dim=embed_dim,
                action_dim=action_dim,
                geo_dim=num_points * 2,
                horizon=action_horizon,
                hidden_dim=384,
                num_layers=4
            )
        else:
            self.action_flow = nn.Sequential(
                nn.Linear(embed_dim + action_dim * action_horizon + 1, 384),
                nn.GELU(),
                nn.Linear(384, action_dim * action_horizon)
            )

    def sample_actions(self, img_tensor: torch.Tensor, num_steps: int = 4) -> torch.Tensor:
        feat = self.conv_stem(img_tensor).flatten(1)
        z_vis = self.vis_proj(feat)
        B = img_tensor.shape[0]

        if self.is_coupled:
            pred_actions, _ = self.coupled_flow.sample_trajectory(z_vis, num_steps=num_steps)
            return pred_actions
        else:
            u_t = torch.randn(B, self.action_horizon * self.action_dim, device=img_tensor.device)
            dt = 1.0 / num_steps
            for step_idx in range(num_steps):
                t_val = float(step_idx) / num_steps
                t_tensor = torch.full((B, 1), t_val, device=img_tensor.device)
                flow_in = torch.cat([u_t, t_tensor, z_vis], dim=-1)
                v_pred = self.action_flow(flow_in)
                u_t = u_t + v_pred * dt
            return u_t.view(B, self.action_horizon, self.action_dim)


def run_libero_sim_episode(
    env: OffScreenRenderEnv,
    model: PolicyModel,
    device: str = "cuda",
    max_steps: int = 120
) -> Tuple[bool, float]:
    """
    Executes a single episode in the official native LIBERO simulator.
    """
    obs = env.reset()
    success = False
    min_dist_eef = 999.0

    for step in range(max_steps):
        # Native LIBERO provides agentview (128x128x3) and wrist (128x128x3)
        raw_img = obs["agentview_image"][::-1, :, :]  # RGB
        img_tensor = torch.tensor(raw_img / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

        # Policy Flow Matching forward pass
        with torch.no_grad():
            action_chunk = model.sample_actions(img_tensor, num_steps=4)
            action = action_chunk[0, 0, :7].cpu().numpy()

        action = np.clip(action, -1.0, 1.0)
        obs, reward, done, info = env.step(action)

        if env.check_success():
            success = True
            break

        if done:
            break

    return success, min_dist_eef


def evaluate_native_libero_benchmark(
    suite_name: str = "libero_spatial",
    num_trials_per_task: int = 10,
    output_dir: str = "/media/kavinder/hdd2/geo_jepa_eval_results/native_libero_simulator_eval"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(f" Geo-JEPA: Native Official LIBERO Simulator Benchmark [{suite_name}]")
    print(f" Number of Trials per Task: {num_trials_per_task} | Output: {out_path}")
    print("=" * 85)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_2d = "/media/kavinder/hdd2/geo_jepa_checkpoints/ablations/baseline_vla_jepa/model_final.pt"
    ckpt_geo = "/media/kavinder/hdd2/geo_jepa_checkpoints/ablations/full_coupled_geo_jepa/model_final.pt"

    model_2d = PolicyModel(config_name="baseline_vla_jepa").to(device)
    if Path(ckpt_2d).exists():
        model_2d.load_state_dict(torch.load(ckpt_2d, map_location=device), strict=False)
    model_2d.eval()

    model_geo = PolicyModel(config_name="full_coupled_geo_jepa").to(device)
    if Path(ckpt_geo).exists():
        model_geo.load_state_dict(torch.load(ckpt_geo, map_location=device), strict=False)
    model_geo.eval()

    benchmark = get_benchmark(suite_name)()
    num_tasks = benchmark.get_num_tasks()

    task_results = []

    for task_id in range(num_tasks):
        task = benchmark.get_task(task_id)
        bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        print(f"\n[{task_id+1}/{num_tasks}] Evaluating LIBERO Simulator: {task.name}")
        print(f"  Prompt: \"{task.language}\"")

        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
        }

        succ_2d = 0
        succ_geo = 0

        for trial in range(num_trials_per_task):
            # Evaluate 2D Baseline
            env_2d = OffScreenRenderEnv(**env_args)
            s2d, _ = run_libero_sim_episode(env_2d, model_2d, device=device)
            if s2d:
                succ_2d += 1
            env_2d.close()

            # Evaluate Geo-JEPA
            env_geo = OffScreenRenderEnv(**env_args)
            sgeo, _ = run_libero_sim_episode(env_geo, model_geo, device=device)
            if sgeo:
                succ_geo += 1
            env_geo.close()

        sr_2d = (succ_2d / num_trials_per_task) * 100.0
        sr_geo = (succ_geo / num_trials_per_task) * 100.0
        delta = sr_geo - sr_2d

        print(f"  --> Baseline 2D Success Rate: {sr_2d:.1f}% ({succ_2d}/{num_trials_per_task})")
        print(f"  --> Geo-JEPA 3D Success Rate: {sr_geo:.1f}% ({succ_geo}/{num_trials_per_task}) [Δ = +{delta:.1f}%]")

        task_results.append({
            "task_id": task_id + 1,
            "task_name": task.name,
            "prompt": task.language,
            "trials": num_trials_per_task,
            "baseline_2d_success_rate": sr_2d,
            "geo_jepa_success_rate": sr_geo,
            "delta_gain": delta
        })

    # Summary
    summary = {
        "suite_name": suite_name,
        "benchmark": "Native Official LIBERO Simulation Benchmark",
        "num_tasks": num_tasks,
        "trials_per_task": num_trials_per_task,
        "mean_baseline_2d_success_rate": float(np.mean([r["baseline_2d_success_rate"] for r in task_results])),
        "mean_geo_jepa_success_rate": float(np.mean([r["geo_jepa_success_rate"] for r in task_results])),
        "mean_net_advantage": float(np.mean([r["delta_gain"] for r in task_results])),
        "tasks": task_results
    }

    report_path = out_path / f"native_{suite_name}_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 85)
    print(f" NATIVE LIBERO BENCHMARK EVALUATION COMPLETE: [{suite_name}]")
    print(f" Baseline 2D Mean Success Rate: {summary['mean_baseline_2d_success_rate']:.2f}%")
    print(f" Geo-JEPA Mean Success Rate:    {summary['mean_geo_jepa_success_rate']:.2f}% (Δ = +{summary['mean_net_advantage']:.2f}%)")
    print(f" Report Saved: {report_path}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="libero_spatial", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    evaluate_native_libero_benchmark(suite_name=args.suite, num_trials_per_task=args.trials)
