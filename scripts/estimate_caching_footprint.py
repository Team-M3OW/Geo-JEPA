"""
Feature Caching Footprint Estimator for Geo-JEPA.

Calculates the exact per-frame and per-clip tensor dimensions, data types,
and projected disk usage across pretraining and fine-tuning datasets.
"""

import numpy as np


def compute_footprint():
    # Sequence parameters
    k = 2  # past context frames
    T = 8  # future prediction frames
    window_len = k + 1 + T  # 11 frames
    
    # Image resolution for VGGT: 518x518
    H, W = 518, 518
    patch_size = 14
    patch_h, patch_w = H // patch_size, W // patch_size  # 37 x 37 = 1369 spatial patches
    num_patches = patch_h * patch_w  # 1369
    
    # Feature dimensions
    vggt_dim = 2048  # Concatenated frame + global token features
    num_track_queries = 64
    
    # Bytes per element
    fp16_bytes = 2
    fp32_bytes = 4
    
    print("=" * 70)
    print(" Geo-JEPA Per-Clip Tensor Footprint Analysis")
    print("=" * 70)
    print(f" Temporal Window: past_k={k}, current=1, future_T={T} (Total {window_len} frames)")
    print(f" Spatial Grid:    {H}x{W} pixels => {patch_h}x{patch_w} = {num_patches} patches")
    print("-" * 70)
    
    # Per single view calculations
    # 1. Backbone latents: [S, num_patches, 2048]
    latent_elements = window_len * num_patches * vggt_dim
    latent_raw_fp16 = latent_elements * fp16_bytes
    
    # 2. Depth maps: [S, H, W, 1]
    depth_elements = window_len * H * W * 1
    depth_raw_fp16 = depth_elements * fp16_bytes
    
    # 3. Canonicalized 3D points: [S, H, W, 3]
    pts_elements = window_len * H * W * 3
    pts_raw_fp16 = pts_elements * fp16_bytes
    
    # 4. Point track displacements: [S-1, num_track_queries, 2]
    track_elements = (window_len - 1) * num_track_queries * 2
    track_raw_fp16 = track_elements * fp16_bytes
    
    # 5. Extrinsics + Intrinsics: [S, 3, 4] + [S, 3, 3]
    pose_elements = window_len * (12 + 9)
    pose_raw_fp32 = pose_elements * fp32_bytes
    
    total_raw_per_view_mb = (latent_raw_fp16 + depth_raw_fp16 + pts_raw_fp16 + track_raw_fp16 + pose_raw_fp32) / (1024 * 1024)
    
    # Compression ratio for smooth neural features is typically ~2.5x to 3.5x with npz zlib
    approx_compressed_per_view_mb = total_raw_per_view_mb / 2.8
    
    print(f" Breakdown (Per View, FP16):")
    print(f"   - Backbone Latents (11 x 1369 x 2048):  {latent_raw_fp16 / (1024*1024):.2f} MB")
    print(f"   - Depth Maps (11 x 518 x 518 x 1):       {depth_raw_fp16 / (1024*1024):.2f} MB")
    print(f"   - Canonical 3D Points (11 x 518 x 518 x 3): {pts_raw_fp16 / (1024*1024):.2f} MB")
    print(f"   - Track Displacements (10 x 64 x 2):      {track_raw_fp16 / 1024:.2f} KB")
    print(f"   - Poses & Intrinsics:                     {pose_raw_fp32 / 1024:.2f} KB")
    print(f" Total Raw Uncompressed / View:              {total_raw_per_view_mb:.2f} MB")
    print(f" Approx. Compressed (.npz) / View:          ~ {approx_compressed_per_view_mb:.2f} MB")
    print("-" * 70)
    
    # Dataset projections
    print(" Projected Dataset Storage Requirements:")
    
    # LIBERO: 500 demos x 2 views (agentview + wrist)
    libero_clips = 500
    libero_views = 2
    libero_gb = (libero_clips * libero_views * approx_compressed_per_view_mb) / 1024
    print(f"   1. LIBERO Benchmark ({libero_clips} demos, 2 views):       ~ {libero_gb:.2f} GB")
    
    # SSv2: 220K clips x 1 view (monocular)
    ssv2_clips = 220_000
    ssv2_views = 1
    # For human video JEPA, if caching only backbone latents (without dense depth/points)
    ssv2_latent_only_mb = (latent_raw_fp16 / 2.8) / (1024 * 1024)
    ssv2_full_tb = (ssv2_clips * ssv2_views * approx_compressed_per_view_mb) / (1024 * 1024)
    ssv2_latent_only_gb = (ssv2_clips * ssv2_views * ssv2_latent_only_mb) / 1024
    print(f"   2. SSv2 Pretraining (220K clips, 1-view):")
    print(f"        - Full (Latents + Dense Points + Depth): ~ {ssv2_full_tb:.2f} TB")
    print(f"        - Latents Only (for L_geo alignment):    ~ {ssv2_latent_only_gb:.2f} GB")
    
    # DROID: 76K trajectories x 2 views
    droid_trajs = 76_000
    droid_views = 2
    droid_full_tb = (droid_trajs * droid_views * approx_compressed_per_view_mb) / (1024 * 1024)
    print(f"   3. DROID Pretraining (76K trajectories, 2 views):   ~ {droid_full_tb:.2f} TB")
    print("=" * 70)
    print(" Storage Capacity on Secondary Disk (/media/kavinder/hdd2): 2.9 TB Free")
    print("=" * 70)


if __name__ == "__main__":
    compute_footprint()
