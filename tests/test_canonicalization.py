"""
Regression Test for Geometric Canonicalization.

Verifies the core property:
For a sequence with camera motion in a static scene:
- Camera-relative point maps drift significantly across frames (nuisance motion).
- Canonicalized point maps remain near-identical (< 1e-5 error) across all frames.
"""

import unittest
import numpy as np
import torch

from geo_jepa.vggt_wrapper.canonicalization import (
    quat_to_rotmat,
    depth_to_camera_coordinates,
    camera_to_world_coordinates,
    canonicalize_point_map,
    compute_point_track_displacements,
)


class TestCanonicalization(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_canonicalization_invariance_under_camera_motion(self):
        """
        Verify that canonicalized point maps eliminate camera motion artifacts for static 3D scenes.
        """
        B, S, H, W = 2, 8, 32, 32
        
        # 1. Create a static 3D world scene
        # 3D points placed in front of camera (Z between 1.0 and 3.0 meters)
        x_grid, y_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, W, device=self.device),
            torch.linspace(-1.0, 1.0, H, device=self.device),
            indexing="xy"
        )
        z_grid = 2.0 + 0.3 * torch.sin(x_grid * 3.0) * torch.cos(y_grid * 3.0)
        static_world_points = torch.stack([x_grid, y_grid, z_grid], dim=-1)  # (H, W, 3)
        static_world_points = static_world_points.unsqueeze(0).unsqueeze(0).expand(B, S, H, W, 3).clone()

        # 2. Simulate camera motion trajectory over S=8 frames:
        # Panning (yaw), tilting (pitch), and 3D translation
        extrinsics_list = []
        for b in range(B):
            extri_b = []
            for t in range(S):
                # Camera angles and translations varying smoothly with time
                yaw = 0.05 * t + 0.02 * b
                pitch = -0.03 * t
                roll = 0.01 * t
                
                # Euler to rotation matrix
                cy, sy = np.cos(yaw), np.sin(yaw)
                cp, sp = np.cos(pitch), np.sin(pitch)
                cr, sr = np.cos(roll), np.sin(roll)
                
                R_np = np.array([
                    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                    [-sp,     cp * sr,                cp * cr]
                ], dtype=np.float32)
                
                # Translation: moving camera left/right, up/down, forward/back
                t_np = np.array([0.15 * t, -0.08 * t, 0.2 * t], dtype=np.float32)
                
                extri_t = np.concatenate([R_np, t_np[:, None]], axis=-1)  # (3, 4)
                extri_b.append(extri_t)
            extrinsics_list.append(np.stack(extri_b, axis=0))
            
        extrinsics = torch.tensor(np.stack(extrinsics_list, axis=0), device=self.device)  # (B, S, 3, 4)

        # 3. Compute camera-relative 3D points: X_cam = R * X_world + t
        R = extrinsics[..., :3, :3]  # (B, S, 3, 3)
        t = extrinsics[..., :3, 3]   # (B, S, 3)
        
        # Multiply: static_world_points @ R^T + t
        R_exp = R.unsqueeze(2).unsqueeze(2)  # (B, S, 1, 1, 3, 3)
        t_exp = t.unsqueeze(2).unsqueeze(2)  # (B, S, 1, 1, 3)
        
        camera_relative_points = torch.matmul(static_world_points.unsqueeze(-2), R_exp.transpose(-1, -2)).squeeze(-2) + t_exp

        # 4. Canonicalize points back into frame 0's coordinate system
        world_points_recovered = camera_to_world_coordinates(camera_relative_points, extrinsics)
        anchor_extri = extrinsics[:, 0]  # Frame 0 extrinsics (B, 3, 4)
        canonicalized_points = canonicalize_point_map(world_points_recovered, anchor_extrinsics=anchor_extri)

        # 5. Verify Drift vs Invariance
        # Camera-relative points should drift significantly from frame 0
        cam_drift = torch.norm(camera_relative_points[:, 1:] - camera_relative_points[:, :1], dim=-1).mean().item()
        
        # Canonicalized points should match frame 0 within numerical precision
        canon_error = torch.norm(canonicalized_points[:, 1:] - canonicalized_points[:, :1], dim=-1).mean().item()

        print(f"\n[Canonicalization Test Results]")
        print(f"  Camera-relative drift across frames: {cam_drift:.4f} meters (demonstrates camera motion artifact)")
        print(f"  Canonicalized point map error:        {canon_error:.8e} meters (near-zero invariant)")

        self.assertGreater(cam_drift, 0.2, "Camera relative points should exhibit noticeable drift due to motion")
        self.assertLess(canon_error, 1e-5, "Canonicalized point maps must be invariant across all camera frames")

    def test_depth_unprojection_roundtrip(self):
        """
        Verify that depth unprojection to 3D camera coordinates followed by camera_to_world produces exact points.
        """
        B, S, H, W = 1, 4, 16, 16
        
        # Synthetic intrinsics: fx=fy=200, cx=W/2, cy=H/2
        intrinsics = torch.zeros(B, S, 3, 3, device=self.device)
        intrinsics[..., 0, 0] = 200.0
        intrinsics[..., 1, 1] = 200.0
        intrinsics[..., 0, 2] = W / 2.0
        intrinsics[..., 1, 2] = H / 2.0
        intrinsics[..., 2, 2] = 1.0

        # Identity extrinsics
        extrinsics = torch.zeros(B, S, 3, 4, device=self.device)
        extrinsics[..., 0, 0] = extrinsics[..., 1, 1] = extrinsics[..., 2, 2] = 1.0

        # Synthetic planar depth
        depth = torch.full((B, S, H, W), 2.5, device=self.device)
        
        pts_cam = depth_to_camera_coordinates(depth, intrinsics)
        pts_world = camera_to_world_coordinates(pts_cam, extrinsics)

        # At pixel center (cx, cy), x and y should be 0, and z should be 2.5
        center_x = pts_world[0, 0, H // 2, W // 2, 0].item()
        center_y = pts_world[0, 0, H // 2, W // 2, 1].item()
        center_z = pts_world[0, 0, H // 2, W // 2, 2].item()

        self.assertAlmostEqual(center_x, 0.0, places=4)
        self.assertAlmostEqual(center_y, 0.0, places=4)
        self.assertAlmostEqual(center_z, 2.5, places=4)

    def test_track_displacements(self):
        """
        Verify consecutive frame displacement calculation for point tracks.
        """
        B, S, N, D = 2, 5, 10, 2
        # Points moving linearly by delta=(1.5, -0.5) per timestep
        delta = torch.tensor([1.5, -0.5], device=self.device)
        t_steps = torch.arange(S, device=self.device).view(1, S, 1, 1)
        base_tracks = torch.randn(B, 1, N, D, device=self.device)
        tracks = base_tracks + t_steps * delta

        displacements = compute_point_track_displacements(tracks)
        
        self.assertEqual(displacements.shape, (B, S - 1, N, D))
        expected_disp = delta.expand_as(displacements)
        self.assertTrue(torch.allclose(displacements, expected_disp, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
