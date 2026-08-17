#!/usr/bin/env python3
"""
Geo-JEPA Master Phase-Wise Extended Training Pipeline Orchestrator.

Sequentially executes the full production-scale training lifecycle:
  Phase 1: Video World-Model Pretraining with Spatial Forcing (10,000 steps)
  Phase 2: Cross-Embodiment Robot Co-Training & Flow Matching (30,000 steps)
  Phase 3: Task-Specific Stabilization & Action Ray Guidance (20,000 steps)
  Phase 4: Official Native LIBERO Simulator Benchmark Evaluation

Logs telemetry locally and saves checkpoints to /media/kavinder/hdd2/geo_jepa_runs/
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON_EXEC = "/home/kavinder/miniconda3/envs/jepa/bin/python"
BASE_DIR = "/home/kavinder/Geo-JEPA"


def run_command(cmd: list, desc: str):
    print("\n" + "=" * 85)
    print(f" >>> [LAUNCHING]: {desc}")
    print(f" >>> CMD: {' '.join(cmd)}")
    print("=" * 85 + "\n")
    
    start_t = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"/home/kavinder/LIBERO:{BASE_DIR}:" + env.get("PYTHONPATH", "")
    
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process.stdout:
        print(line, end="", flush=True)
        
    process.wait()
    elapsed = time.time() - start_t
    
    if process.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {process.returncode} in {elapsed:.1f}s: {desc}")
        sys.exit(process.returncode)
    else:
        print(f"\n[SUCCESS] Completed {desc} in {elapsed:.1f}s ({elapsed/60.0:.2f} min)\n")


def main():
    parser = argparse.ArgumentParser(description="Geo-JEPA Phase-Wise Extended Training Pipeline")
    parser.add_argument("--start_phase", type=int, default=1, choices=[1, 2, 3, 4], help="Phase to start from")
    parser.add_argument("--p1_steps", type=int, default=10000, help="Phase 1 video pretraining steps")
    parser.add_argument("--p2_steps", type=int, default=30000, help="Phase 2 robot co-training steps")
    parser.add_argument("--p3_steps", type=int, default=20000, help="Phase 3 task fine-tuning steps")
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging")
    args = parser.parse_args()

    wandb_flag = "--wandb" if args.wandb else "--no_wandb"

    print("=" * 85)
    print(" GEO-JEPA MASTER PHASE-WISE EXTENDED TRAINING PIPELINE")
    print(f" Step Budget: Phase 1={args.p1_steps} | Phase 2={args.p2_steps} | Phase 3={args.p3_steps}")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # PHASE 1: Video-JEPA Pretraining with Spatial Forcing
    # -------------------------------------------------------------------------
    if args.start_phase <= 1:
        cmd_p1 = [
            PYTHON_EXEC,
            f"{BASE_DIR}/scripts/train_phase1_video.py",
            "--config", f"{BASE_DIR}/configs/phase1_video_pretrain.yaml",
            "--steps", str(args.p1_steps),
            wandb_flag
        ]
        run_command(cmd_p1, f"Phase 1: Video World-Model Pretraining ({args.p1_steps} Steps)")

    # -------------------------------------------------------------------------
    # PHASE 2: Robot Co-Training with Coupled Flow Matching
    # -------------------------------------------------------------------------
    if args.start_phase <= 2:
        cmd_p2 = [
            PYTHON_EXEC,
            f"{BASE_DIR}/scripts/train_phase2_robot_cotrain.py",
            "--config", f"{BASE_DIR}/configs/phase2_robot_cotrain.yaml",
            "--steps", str(args.p2_steps),
            wandb_flag
        ]
        run_command(cmd_p2, f"Phase 2: Robot Co-Training & Flow Matching ({args.p2_steps} Steps)")

    # -------------------------------------------------------------------------
    # PHASE 3: Task-Specific Fine-Tuning with Action Rays & Jitter Augmentation
    # -------------------------------------------------------------------------
    if args.start_phase <= 3:
        cmd_p3 = [
            PYTHON_EXEC,
            f"{BASE_DIR}/scripts/train_libero_spatial.py",
            "--config", f"{BASE_DIR}/configs/libero_spatial_full_geo_jepa.yaml",
            "--steps", str(args.p3_steps),
            wandb_flag
        ]
        run_command(cmd_p3, f"Phase 3: LIBERO-Spatial Fine-Tuning & Stabilization ({args.p3_steps} Steps)")

    # -------------------------------------------------------------------------
    # PHASE 4: Official Native LIBERO Simulator Closed-Loop Benchmark
    # -------------------------------------------------------------------------
    if args.start_phase <= 4:
        cmd_p4 = [
            PYTHON_EXEC,
            f"{BASE_DIR}/scripts/eval_native_libero_simulator.py",
            "--suite", "libero_spatial",
            "--trials", "10"
        ]
        run_command(cmd_p4, "Phase 4: Official Native LIBERO Closed-Loop Simulator Benchmark")

    print("\n" + "=" * 85)
    print(" ALL TRAINING PHASES & EVALUATION SUITES SUCCESSFULLY COMPLETED!")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
