"""
Geo-JEPA Dense 3D Action Ray Bundle & Flow Field Projection Head.

Projects latent action tokens <action_i> into a DENSE 3D Volumetric Ray Bundle:
1. Dense 3D Streamline Bundle: N=32 continuous line-of-sight vectors spanning the grasp frustum
2. Dense 3D Point-Track Displacement Field: (N=64, 3) spatial velocity vectors
3. Continuous Metric Distance & Aperture Field

Loss:
  L_dense_bundle = L_streamlines(32 rays) + 0.5 * L_flow_field + 0.2 * L_aperture
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseRayBundleProjector(nn.Module):
    """
    Projects latent action tokens into a Dense 3D Ray Bundle & Vector Field.
    """

    def __init__(
        self,
        action_dim: int = 1024,
        hidden_dim: int = 512,
        num_dense_rays: int = 32,
        num_track_points: int = 64
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_dense_rays = num_dense_rays
        self.num_track_points = num_track_points

        # 1. Dense 3D Ray Bundle MLP (N=32 unit 3D direction vectors)
        self.dense_rays_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_dense_rays * 3)  # 32 * 3 = 96 values
        )

        # 2. Dense 3D Point-Track Velocity Field MLP (N=64 3D vectors)
        self.dense_flow_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_track_points * 3)
        )

        # 3. Metric Distance Field & Aperture Head
        self.dist_aperture_mlp = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 4),  # [min_dist, mean_dist, max_dist, aperture]
            nn.Softplus()
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
            - dense_rays: (B, 32, 3) normalized dense 3D direction streamline bundle
            - flow_field: (B, 64, 3) dense 3D spatial velocity vectors
            - distances: (B, 3) [min_dist, mean_dist, max_dist] in meters
            - aperture: (B, 1) gripper aperture opening in meters
        """
        if action_tokens.dim() == 3:
            action_feat = action_tokens.mean(dim=1)
        else:
            action_feat = action_tokens

        B = action_feat.shape[0]

        # 1. Dense 32-Ray Bundle
        raw_rays = self.dense_rays_mlp(action_feat).view(B, self.num_dense_rays, 3)
        dense_rays = F.normalize(raw_rays, p=2, dim=-1)

        # 2. Dense 64-Point 3D Flow Field
        flow_field = self.dense_flow_mlp(action_feat).view(B, self.num_track_points, 3)

        # 3. Distance & Aperture
        stats = self.dist_aperture_mlp(action_feat)
        distances = stats[:, :3]
        aperture = stats[:, 3:4] * 0.10  # Clamped to 10cm max

        return {
            "dense_rays": dense_rays,
            "flow_field": flow_field,
            "distances": distances,
            "aperture": aperture
        }

    def compute_dense_loss(
        self,
        pred_output: Dict[str, torch.Tensor],
        gt_dense_rays: torch.Tensor,
        gt_flow_field: torch.Tensor,
        gt_distances: torch.Tensor,
        gt_aperture: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computes dense optimal transport & ray-bundle loss.
        """
        # Cosine direction loss over all 32 dense streamlines
        cos_sim = (pred_output["dense_rays"] * gt_dense_rays).sum(dim=-1)
        loss_rays = (1.0 - cos_sim).mean()

        # Dense flow field MSE
        loss_flow = F.mse_loss(pred_output["flow_field"], gt_flow_field)

        # Distance & aperture loss
        loss_dist = F.smooth_l1_loss(pred_output["distances"], gt_distances)
        loss_aperture = F.smooth_l1_loss(pred_output["aperture"], gt_aperture)

        total_loss = loss_rays + 0.5 * loss_flow + 0.3 * loss_dist + 0.2 * loss_aperture

        return {
            "loss_dense_total": total_loss,
            "loss_rays": loss_rays,
            "loss_flow": loss_flow,
            "mean_cos_sim": cos_sim.mean()
        }
