"""
Geo-JEPA Action-Grounded 3D Ray-Bundle Projection Head.

Supervises latent action tokens <action_i> to directly predict:
1. Normalized 3D directional reach rays (r_hat in R^3)
2. Metric Euclidean distance to target (d_hat in R^1)

Loss:
  L_ray = (1 - cosine(r_hat, r_gt)) + SmoothL1(d_hat, d_gt)
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionRayProjector(nn.Module):
    """
    Projects latent action tokens into explicit 3D line-of-sight ray bundles.
    """

    def __init__(self, action_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 3D Directional Ray Head
        self.ray_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3)  # Unit 3D direction vector (rx, ry, rz)
        )

        # Metric Distance Head
        self.dist_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Positive distance in meters
        )

    def forward(self, action_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            action_tokens: (B, N_tokens, D) or (B, D)
            
        Returns:
            ray_dir: (B, 3) normalized unit 3D direction vectors
            ray_dist: (B, 1) predicted metric distance to target in meters
        """
        if action_tokens.dim() == 3:
            # Pool across action tokens
            action_feat = action_tokens.mean(dim=1)
        else:
            action_feat = action_tokens

        raw_ray = self.ray_mlp(action_feat)
        ray_dir = F.normalize(raw_ray, p=2, dim=-1)  # (B, 3)
        ray_dist = self.dist_mlp(action_feat)        # (B, 1)

        return ray_dir, ray_dist

    def compute_ray_loss(
        self,
        pred_ray: torch.Tensor,
        pred_dist: torch.Tensor,
        target_pos_3d: torch.Tensor,
        gripper_pos_3d: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computes L_ray against ground-truth 3D relative vectors.
        """
        # Ground truth delta vector
        gt_vector = target_pos_3d - gripper_pos_3d  # (B, 3)
        gt_dist = torch.norm(gt_vector, p=2, dim=-1, keepdim=True).clamp(min=1e-6)  # (B, 1)
        gt_dir = gt_vector / gt_dist  # (B, 3)

        # Directional cosine distance loss
        cos_sim = (pred_ray * gt_dir).sum(dim=-1, keepdim=True)
        loss_dir = (1.0 - cos_sim).mean()

        # Distance regression loss
        loss_dist = F.smooth_l1_loss(pred_dist, gt_dist)

        total_ray_loss = loss_dir + 0.5 * loss_dist

        return {
            "loss_ray_total": total_ray_loss,
            "loss_ray_dir": loss_dir,
            "loss_ray_dist": loss_dist,
            "cos_sim": cos_sim.mean()
        }
