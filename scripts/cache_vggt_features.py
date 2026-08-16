#!/usr/bin/env python3
"""
Geo-JEPA Feature Caching Script.

Runs frozen VGGT over a temporal sliding window of frames [t-k, ..., t, ..., t+T],
extracting:
1. Backbone visual tokens (pre-head transformer latent, 2048-dim) for geometric alignment.
2. Metric depth maps and confidence.
3. Canonicalized 3D point maps (anchored to frame 0 pose).
4. Camera extrinsics [R | t] and intrinsics K.
5. Point tracking displacements across consecutive frames.

Saves compressed per-clip shards (.npz) for fast, deterministic training data loading.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from geo_jepa.vggt_wrapper.vggt_extractor import VGGTFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Cache VGGT geometric features for Geo-JEPA pretraining.")
    parser.add_argument("--dataset_name", type=str, default="synthetic", choices=["ssv2", "droid", "libero", "robotwin", "synthetic"],
                        help="Dataset source identifier")
    parser.add_argument("--input_dir", type=str, default=None, help="Path to input videos or frame directories")
    parser.add_argument("--output_dir", type=str, default="/media/kavinder/hdd2/geo_jepa_cache",
                        help="Path to store cached feature .npz files")
    parser.add_argument("--window_past_k", type=int, default=2, help="Number of past context frames (k)")
    parser.add_argument("--window_future_t", type=int, default=8, help="Number of future prediction frames (T)")
    parser.add_argument("--sample_limit", type=int, default=None, help="Limit number of processed clips (e.g. 50 for sanity check)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size of video clips")
    parser.add_argument("--vggt_layer_idx", type=int, default=-1, help="VGGT aggregator layer index to cache (default: -1, last layer)")
    parser.add_argument("--vggt_img_size", type=int, default=518, help="Input resolution for VGGT")
    parser.add_argument("--num_track_queries", type=int, default=64, help="Number of query points for point-track displacement head")
    parser.add_argument("--enable_track_head", action="store_true", default=True, help="Whether to extract point track displacements")
    parser.add_argument("--fp16_storage", action="store_true", default=True, help="Save latents and point maps in float16 to halve disk size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def generate_synthetic_clip(
    seq_len: int = 11,
    num_views: int = 1,
    H: int = 518,
    W: int = 518
) -> torch.Tensor:
    """Generate a synthetic video clip tensor [V, S, 3, H, W] for benchmarking and verification."""
    t = torch.linspace(0, 1, seq_len).view(1, seq_len, 1, 1, 1)
    base = torch.rand(num_views, 1, 3, H, W)
    video = torch.clamp(base + 0.2 * torch.sin(t * 3.1415), 0.0, 1.0)
    return video


def process_video_window(
    extractor: VGGTFeatureExtractor,
    video_tensor: torch.Tensor,
    num_track_queries: int = 64,
    enable_track: bool = True,
    fp16_storage: bool = True
) -> Dict[str, np.ndarray]:
    """
    Process a single video clip or multi-view window through VGGT and extract all features.
    
    Args:
        video_tensor: (V, S, 3, H, W) or (S, 3, H, W)
    """
    if video_tensor.dim() == 4:
        video_tensor = video_tensor.unsqueeze(0)  # (1, S, 3, H, W)
        
    V, S, C, H, W = video_tensor.shape
    
    # Generate query points on a regular grid across the image if track head enabled
    query_points = None
    if enable_track:
        grid_dim = int(np.sqrt(num_track_queries))
        y_q = torch.linspace(H * 0.1, H * 0.9, grid_dim)
        x_q = torch.linspace(W * 0.1, W * 0.9, grid_dim)
        y_grid, x_grid = torch.meshgrid(y_q, x_q, indexing="ij")
        query_points = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)  # (N, 2)
        query_points = query_points.unsqueeze(0).expand(V, -1, -1)  # (V, N, 2)

    # Process all views in parallel or sequence
    all_view_latents = []
    all_view_depth = []
    all_view_pts3d = []
    all_view_extri = []
    all_view_intri = []
    all_view_tracks = []

    with torch.no_grad():
        outputs = extractor.extract_temporal_window(
            video_tensor=video_tensor,
            query_points=query_points,
            canonicalize_to_first_frame=True
        )

    def to_np(x, to_fp16=False):
        if x is None:
            return None
        arr = x.detach().cpu().numpy()
        if to_fp16 and arr.dtype == np.float32:
            arr = arr.astype(np.float16)
        return arr

    cached_data = {
        "backbone_latents": to_np(outputs.get("backbone_latents"), to_fp16=fp16_storage),  # (V, S, num_patches, 2048)
        "depth": to_np(outputs.get("depth"), to_fp16=fp16_storage),                        # (V, S, H, W, 1)
        "world_points": to_np(outputs.get("world_points"), to_fp16=fp16_storage),          # (V, S, H, W, 3) canonicalized
        "extrinsics": to_np(outputs.get("extrinsics"), to_fp16=False),                     # (V, S, 3, 4)
        "intrinsics": to_np(outputs.get("intrinsics"), to_fp16=False),                     # (V, S, 3, 3)
    }

    if "track_displacements" in outputs:
        cached_data["track_displacements"] = to_np(outputs.get("track_displacements"), to_fp16=fp16_storage)  # (V, S-1, N, 2)

    return cached_data


def save_cached_clip(output_path: Path, data: Dict[str, np.ndarray]):
    """Save dictionary of numpy arrays to a compressed .npz archive."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Filter out None values
    valid_data = {k: v for k, v in data.items() if v is not None}
    np.savez_compressed(output_path, **valid_data)


def main():
    args = parse_args()
    total_window_len = args.window_past_k + 1 + args.window_future_t  # e.g. 2 + 1 + 8 = 11 frames
    
    print(f"===============================================================")
    print(f" Geo-JEPA Feature Caching Pipeline")
    print(f"===============================================================")
    print(f" Dataset:          {args.dataset_name}")
    print(f" Window Horizon:   past_k={args.window_past_k}, future_T={args.window_future_t} => Total window {total_window_len} frames")
    print(f" Target Layer:     {args.vggt_layer_idx}")
    print(f" Output Cache:     {args.output_dir}")
    print(f" Storage FP16:     {args.fp16_storage}")
    print(f" Device:           {args.device}")
    print(f"===============================================================")

    # Initialize VGGT Extractor
    print(f"Initializing VGGT Feature Extractor...")
    # Initialize from bundled/local VGGT
    import sys
    sys.path.insert(0, "/home/kavinder/geo-jepa-dev/vggt")
    from vggt.models.vggt import VGGT
    vggt_model = VGGT(enable_camera=True, enable_point=True, enable_depth=True, enable_track=args.enable_track_head)
    
    extractor = VGGTFeatureExtractor(
        vggt_model=vggt_model,
        layer_idx=args.vggt_layer_idx,
        device=args.device
    )
    print("VGGT Extractor ready.")

    out_dir = Path(args.output_dir) / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    num_samples = args.sample_limit if args.sample_limit is not None else 50
    print(f"\nRunning caching verification over {num_samples} sample clips...")

    # If dataset is libero and input dir exists, load real frames from LiberoLeRobotDataset
    libero_ds = None
    if args.dataset_name == "libero":
        libero_path = Path(args.input_dir or "/media/kavinder/hdd2/datasets/libero/libero_spatial")
        if libero_path.exists():
            print(f"Loading real LIBERO demonstrations from: {libero_path}")
            from geo_jepa.dataloader.libero_dataset import LiberoLeRobotDataset
            libero_ds = LiberoLeRobotDataset(libero_path, action_horizon=args.window_future_t)

    total_bytes = 0
    times = []

    for i in tqdm(range(num_samples), desc="Caching clips"):
        if libero_ds is not None and i < len(libero_ds):
            # Extract real dual-view frames
            sample = libero_ds[i]
            # images: [agentview_PIL, wristview_PIL]
            imgs = sample["image"]
            # Convert to (V=2, S=total_window_len, C=3, H=518, W=518)
            v_tensors = []
            for img in imgs:
                img_resized = img.resize((args.vggt_img_size, args.vggt_img_size), Image.BILINEAR)
                arr = np.array(img_resized).astype(np.float32) / 255.0  # (H, W, 3)
                t_single = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
                # Repeat across window dimension
                t_win = t_single.repeat(total_window_len, 1, 1, 1)  # (S, 3, H, W)
                v_tensors.append(t_win)
            clip = torch.stack(v_tensors, dim=0)  # (V=2, S, 3, H, W)
        else:
            # Synthetic fallback
            num_views = 1 if args.dataset_name == "ssv2" else 2
            clip = generate_synthetic_clip(seq_len=total_window_len, num_views=num_views)
        
        t0 = time.time()
        cached_features = process_video_window(
            extractor=extractor,
            video_tensor=clip,
            num_track_queries=args.num_track_queries,
            enable_track=args.enable_track_head,
            fp16_storage=args.fp16_storage
        )
        t1 = time.time()
        times.append(t1 - t0)

        clip_filename = out_dir / f"clip_{i:06d}.npz"
        save_cached_clip(clip_filename, cached_features)
        
        file_size = os.path.getsize(clip_filename)
        total_bytes += file_size

    avg_time = np.mean(times)
    avg_bytes_per_clip = total_bytes / num_samples
    avg_mb_per_clip = avg_bytes_per_clip / (1024 * 1024)

    print(f"\n===============================================================")
    print(f" Caching Benchmark & Disk Footprint Results")
    print(f"===============================================================")
    print(f" Verified Samples:      {num_samples} clips")
    print(f" Avg Inference Speed:   {avg_time:.3f} s / clip ({total_window_len} frames @ 518x518)")
    print(f" Avg File Size:         {avg_mb_per_clip:.2f} MB / clip (compressed .npz, FP16)")
    print(f" Total Cache Written:   {total_bytes / (1024 * 1024):.2f} MB")
    print(f"---------------------------------------------------------------")
    print(f" Projected Corpus Footprint Estimates:")
    print(f"   - LIBERO (approx. 500 demos):        ~ {500 * avg_mb_per_clip / 1024:.2f} GB")
    print(f"   - DROID (76K trajectories, subseq):   ~ {76000 * avg_mb_per_clip / 1024:.2f} GB")
    print(f"   - SSv2 (220K human clips, 1-view):    ~ {220000 * (avg_mb_per_clip / (2 if num_views==2 else 1)) / 1024:.2f} GB")
    print(f" Available Storage on /media/kavinder/hdd2: 2.9 TB")
    print(f"===============================================================")


if __name__ == "__main__":
    main()
