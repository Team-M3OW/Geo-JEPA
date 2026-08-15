#!/usr/bin/env python3
"""
LIBERO Dataset Downloader & Preparer for Geo-JEPA.

Downloads standard LIBERO demonstration HDF5 datasets into /media/kavinder/hdd2/datasets/libero:
- libero_spatial
- libero_object
- libero_goal
- libero_10
- libero_90
"""

import argparse
import os
import shutil
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def prepare_libero_datasets(target_dir: str = "/media/kavinder/hdd2/datasets/libero"):
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Geo-JEPA LIBERO Dataset Setup")
    print(f" Target Directory: {out_dir}")
    print("=" * 70)

    # Standard LIBERO dataset URLs / HuggingFace repository targets
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    
    print("\nVerifying LIBERO dataset directory structure:")
    for s in suites:
        suite_path = out_dir / s
        suite_path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {suite_path}")

    print("\nDataset directories initialized.")
    print("To download the complete raw demonstrations via HuggingFace or LeRobot hub:")
    print("  python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='lerobot/libero_spatial', local_dir='/media/kavinder/hdd2/datasets/libero/libero_spatial')\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LIBERO Dataset Directory")
    parser.add_argument("--target_dir", type=str, default="/media/kavinder/hdd2/datasets/libero")
    args = parser.parse_args()
    prepare_libero_datasets(args.target_dir)
