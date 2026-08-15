"""
Geometric Canonicalization Module for Geo-JEPA.

This module provides utilities to:
1. Decode camera extrinsics and intrinsics from VGGT pose encodings.
2. Unproject dense depth maps to 3D camera coordinate point maps.
3. Canonicalize 3D point maps across multi-frame / multi-view sequences into
   a shared anchor frame (e.g. frame 0 or a fixed world frame).
4. Compute consecutive temporal point-track displacements.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions [w, x, y, z] to 3x3 rotation matrices.
    
    Args:
        quat: Tensor of shape (..., 4) with quaternion [w, x, y, z] or [x, y, z, w].
              VGGT convention is [w, x, y, z].
              
    Returns:
        Tensor of shape (..., 3, 3) representing rotation matrices.
    """
    quat = F.normalize(quat, p=2, dim=-1)
    w, x, y, z = quat.unbind(dim=-1)
    
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1)
    
    return torch.stack([row0, row1, row2], dim=-2)


def decode_vggt_pose_encoding(
    pose_enc: torch.Tensor,
    image_size_hw: Tuple[int, int] = (518, 518)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode VGGT 9-dimensional pose encoding into camera extrinsics [R | t] (3x4) and intrinsics K (3x3).
    
    VGGT pose encoding format (absT_quaR_FoV):
        - pose_enc[..., :3]: Absolute translation vector t (camera from world or world from camera)
        - pose_enc[..., 3:7]: Quaternion rotation [w, x, y, z]
        - pose_enc[..., 7]: Field of view height (fov_h)
        - pose_enc[..., 8]: Field of view width (fov_w)
        
    Args:
        pose_enc: Tensor of shape (..., 9) or (B, S, 9)
        image_size_hw: (Height, Width) in pixels
        
    Returns:
        extrinsics: Tensor of shape (..., 3, 4) with [R | t]
        intrinsics: Tensor of shape (..., 3, 3)
    """
    H, W = image_size_hw
    t = pose_enc[..., :3]  # (..., 3)
    quat = pose_enc[..., 3:7]  # (..., 4)
    fov_h = pose_enc[..., 7]  # (...)
    fov_w = pose_enc[..., 8]  # (...)
    
    R = quat_to_rotmat(quat)  # (..., 3, 3)
    extrinsics = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # (..., 3, 4)
    
    # Compute focal lengths from FOV: f = (size / 2) / tan(fov / 2)
    eps = 1e-6
    tan_fov_h = torch.tan(torch.clamp(fov_h / 2.0, min=eps, max=np.pi / 2.0 - eps))
    tan_fov_w = torch.tan(torch.clamp(fov_w / 2.0, min=eps, max=np.pi / 2.0 - eps))
    
    fy = (H / 2.0) / tan_fov_h
    fx = (W / 2.0) / tan_fov_w
    cx = torch.full_like(fx, W / 2.0)
    cy = torch.full_like(fy, H / 2.0)
    
    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)
    
    k_row0 = torch.stack([fx, zeros, cx], dim=-1)
    k_row1 = torch.stack([zeros, fy, cy], dim=-1)
    k_row2 = torch.stack([zeros, zeros, ones], dim=-1)
    intrinsics = torch.stack([k_row0, k_row1, k_row2], dim=-2)  # (..., 3, 3)
    
    return extrinsics, intrinsics


def depth_to_camera_coordinates(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Unproject 2D depth map into 3D camera coordinates (X_cam, Y_cam, Z_cam).
    
    Args:
        depth: Tensor of shape (..., H, W) or (..., H, W, 1)
        intrinsics: Tensor of shape (..., 3, 3)
        eps: Minimum valid depth threshold
        
    Returns:
        points_cam: Tensor of shape (..., H, W, 3) containing metric 3D camera coordinates
    """
    if depth.dim() > 2 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
        
    *batch_dims, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    
    # Generate pixel coordinate grid (u, v)
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )  # (H, W)
    
    # Reshape for broadcasting
    for _ in batch_dims:
        v_grid = v_grid.unsqueeze(0)
        u_grid = u_grid.unsqueeze(0)
        
    fx = intrinsics[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = intrinsics[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1).unsqueeze(-1)
    
    z = torch.clamp(depth, min=eps)
    x = (u_grid - cx) * z / fx
    y = (v_grid - cy) * z / fy
    
    points_cam = torch.stack([x, y, z], dim=-1)  # (..., H, W, 3)
    return points_cam


def camera_to_world_coordinates(
    points_cam: torch.Tensor,
    extrinsics: torch.Tensor
) -> torch.Tensor:
    """
    Transform 3D camera coordinates to world coordinates.
    OpenCV convention: X_cam = R * X_world + t  ==>  X_world = R^T * (X_cam - t)
    
    Args:
        points_cam: Tensor of shape (..., H, W, 3) or (..., N, 3)
        extrinsics: Tensor of shape (..., 3, 4) with [R | t]
        
    Returns:
        points_world: Tensor of shape (..., H, W, 3) or (..., N, 3)
    """
    R = extrinsics[..., :3, :3]  # (..., 3, 3)
    t = extrinsics[..., :3, 3]   # (..., 3)
    
    # Expand t to match spatial dims of points_cam
    spatial_ndim = points_cam.dim() - extrinsics.dim() + 1
    t_expanded = t
    for _ in range(spatial_ndim):
        t_expanded = t_expanded.unsqueeze(-2)
        
    diff = points_cam - t_expanded  # (..., [spatial], 3)
    
    # We want: X_world = R^T @ diff = (diff^T @ R)^T = diff @ R
    # Using einsum for arbitrary spatial dims:
    if points_cam.dim() == 5:  # (B, S, H, W, 3)
        points_world = torch.einsum("bshwi,bsij->bshwj", diff, R)
    elif points_cam.dim() == 4:  # (B, S, N, 3) or (S, H, W, 3)
        points_world = torch.einsum("bsni,bsij->bsnj", diff, R)
    else:
        points_world = torch.einsum("...i,...ij->...j", diff, R)
        
    return points_world


def canonicalize_point_map(
    points_world: torch.Tensor,
    anchor_extrinsics: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Canonicalize 3D world points into the reference coordinate frame of the anchor frame (default: frame 0).
    
    If anchor_extrinsics [R_0 | t_0] is provided:
        X_canon = R_0 * X_world + t_0
    If anchor_extrinsics is None (world frame as reference):
        X_canon = X_world
        
    Args:
        points_world: Tensor of shape (B, S, H, W, 3)
        anchor_extrinsics: Optional tensor of shape (B, 3, 4) representing frame 0 extrinsics
        
    Returns:
        points_canon: Canonicalized point map of shape (B, S, H, W, 3)
    """
    if anchor_extrinsics is None:
        return points_world
        
    R_0 = anchor_extrinsics[..., :3, :3]  # (B, 3, 3)
    t_0 = anchor_extrinsics[..., :3, 3]   # (B, 3)
    
    # Expand t_0 for (S, H, W)
    t_0_exp = t_0.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 1, 3)
    
    # X_canon = (points_world @ R_0^T) + t_0
    if points_world.dim() == 5:
        # points_world: (B, S, H, W, 3), R_0: (B, 3, 3)
        # We want R_0 @ points_world => points_world @ R_0^T
        rot = torch.einsum("bshwi,bji->bshwj", points_world, R_0)
    else:
        rot = torch.einsum("...i,bji->...j", points_world, R_0)
        
    return rot + t_0_exp


def compute_point_track_displacements(
    point_tracks: torch.Tensor
) -> torch.Tensor:
    """
    Compute frame-to-frame displacement vectors for tracked points across a temporal window.
    
    Args:
        point_tracks: Tensor of shape (B, S, N, D) where D=2 (pixel tracks) or D=3 (3D canonical tracks)
                      B: batch size, S: sequence length, N: number of query points
                      
    Returns:
        displacements: Tensor of shape (B, S - 1, N, D) where displacements[:, t] = tracks[:, t+1] - tracks[:, t]
    """
    return point_tracks[:, 1:, ...] - point_tracks[:, :-1, ...]
