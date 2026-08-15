#!/usr/bin/env python3
"""
LIBERO Dataset Downloader from Hugging Face Hub.

Downloads the standard 4 LIBERO benchmark suites to /media/kavinder/hdd2/datasets/libero:
- libero_spatial
- libero_object
- libero_goal
- libero_10
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download


def download_libero_suites(target_root: str = "/media/kavinder/hdd2/datasets/libero"):
    root = Path(target_root)
    root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Downloading LIBERO Benchmark Datasets to:", root)
    print("=" * 70)

    suites = {
        "libero_spatial": "lerobot/libero_spatial",
        "libero_object": "lerobot/libero_object",
        "libero_goal": "lerobot/libero_goal",
        "libero_10": "lerobot/libero_10",
    }

    for name, repo_id in suites.items():
        dest = root / name
        print(f"\n[+] Fetching {name} from {repo_id}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(dest),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"  [OK] Successfully downloaded {name} to {dest}")
        except Exception as e:
            print(f"  [!] Note: {repo_id} download returned: {e}")

    print("\n" + "=" * 70)
    print(" LIBERO Datasets Download Process Completed!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LIBERO benchmark datasets")
    parser.add_argument("--target_root", type=str, default="/media/kavinder/hdd2/datasets/libero")
    args = parser.parse_args()
    download_libero_suites(args.target_root)
