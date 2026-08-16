"""
Geo-JEPA Multi-Ray 3D Grasp Bundle Projection Head.

Projects latent action tokens <action_i> into a full 3D Volumetric Grasp Bundle:
1. Left Finger Contact Ray (r_left in R^3, d_left in R^+)
2. Right Finger Contact Ray (r_right in R^3, d_right in R^+)
3. Central Palm Approach Ray (r_palm in R^3, d_palm in R^+)
4. Volumetric Affordance Cone (4 boundary envelope rays in R^(4 x 3))
5. Predicted Gripper Aperture Opening (w in mm)

Loss:
  L_ray_bundle = L_dir(left, right, palm) + 0.5 * L_dist + 0.2 * L_aperture
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiRayGraspBundleProjector(nn.Module):
    """
    Projects latent action tokens into a complete 3D Multi-Ray Grasp Bundle.
    """

    def __init__(self, action_dim: int = 1024, hidden_dim: int = 512, num_cone_rays: int = 4):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_cone_rays = num_cone_rays

        # 1. Multi-Ray Direction Head: Left, Palm, Right + Cone boundary rays
        # Total rays = 3 (core) + num_cone_rays (envelope)
        self.total_rays = 3 + num_cone_rays
        self.ray_bundle_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.total_rays * 3)  # (3 + 4) * 3 = 21 values
        )

        # 2. Metric Distances Head (Left, Palm, Right in meters)
        self.dist_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Softplus()  # Positive distances in meters
        )

        # 3. Gripper Aperture Opening Head (Aperture in meters, e.g. 0 to 0.10m)
        self.aperture_mlp = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Normalized [0, 1] mapped to 0..100mm
        )

    def forward(
        self,
        action_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            action_tokens: (B, N_tokens, D) or (B, D)
            
        Returns:
            Dictionary containing:
            - ray_left: (B, 3) normalized unit vector to left grasp contact
            - ray_right: (B, 3) normalized unit vector to right grasp contact
            - ray_palm: (B, 3) normalized unit vector along palm approach axis
            - ray_cone: (B, 4, 3) normalized boundary rays defining grasp frustum
            - distances: (B, 3) metric distances [d_left, d_palm, d_right]
            - aperture: (B, 1) predicted gripper opening in meters
        """
        if action_tokens.dim() == 3:
            action_feat = action_tokens.mean(dim=1)
        else:
            action_feat = action_tokens

        B = action_feat.shape[0]

        # Predict raw ray bundle and normalize
        raw_rays = self.ray_bundle_mlp(action_feat).view(B, self.total_rays, 3)
        norm_rays = F.normalize(raw_rays, p=2, dim=-1)

        ray_left = norm_rays[:, 0]
        ray_palm = norm_rays[:, 1]
        ray_right = norm_rays[:, 2]
        ray_cone = norm_rays[:, 3:]  # (B, 4, 3)

        distances = self.dist_mlp(action_feat)  # (B, 3)
        aperture = self.aperture_mlp(action_feat) * 0.10  # 0 to 10 cm max opening

        return {
            "ray_left": ray_left,
            "ray_palm": ray_palm,
            "ray_right": ray_right,
            "ray_cone": ray_cone,
            "distances": distances,
            "aperture": aperture
        }

    def compute_bundle_loss(
        self,
        pred_bundle: Dict[str, torch.Tensor],
        gt_left_contact: torch.Tensor,
        gt_right_contact: torch.Tensor,
        gt_target_center: torch.Tensor,
        gt_gripper_pos: torch.Tensor,
        gt_aperture: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computes multi-ray alignment loss against 3D contact ground truth.
        """
        # Ground truth delta vectors
        v_left = gt_left_contact - gt_gripper_pos
        v_right = gt_right_contact - gt_gripper_pos
        v_palm = gt_target_center - gt_gripper_pos

        d_left = torch.norm(v_left, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        d_right = torch.norm(v_right, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        d_palm = torch.norm(v_palm, p=2, dim=-1, keepdim=True).clamp(min=1e-6)

        gt_dir_left = v_left / d_left
        gt_dir_right = v_right / d_right
        gt_dir_palm = v_palm / d_palm
        gt_dists = torch.cat([d_left, d_palm, d_right], dim=-1)

        # Direction Cosine Losses
        cos_left = (pred_bundle["ray_left"] * gt_dir_left).sum(dim=-1).mean()
        cos_right = (pred_bundle["ray_right"] * gt_dir_right).sum(dim=-1).mean()
        cos_palm = (pred_bundle["ray_palm"] * gt_dir_palm).sum(dim=-1).mean()

        loss_dir = (1.0 - cos_left) + (1.0 - cos_right) + (1.0 - cos_palm)

        # Distance Regression Loss
        loss_dist = F.smooth_l1_loss(pred_bundle["distances"], gt_dists)

        # Aperture Width Loss
        loss_aperture = F.smooth_l1_loss(pred_bundle["aperture"], gt_aperture)

        total_loss = loss_dir + 0.5 * loss_dist + 0.2 * loss_aperture

        return {
            "loss_bundle_total": total_loss,
            "loss_dir": loss_dir,
            "loss_dist": loss_dist,
            "loss_aperture": loss_aperture,
            "mean_cos_sim": (cos_left + cos_right + cos_palm) / 3.0
        }
