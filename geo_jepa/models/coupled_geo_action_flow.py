"""
Geo-JEPA Coupled Geometric-Action Joint Flow Head.

Unifies robot motor trajectory generation (a) and 3D physical point dynamics (Δp)
into a SINGLE joint vector field over the product manifold:

  u = [ a , Δp ] in R^(H x (D_action + D_geo))

Trained with a single unified Optimal Transport Flow-Matching objective:
  L_CoupledFlow = E [ || v_θ( u_t, t, c ) - (u_1 - u_0) ||² ]
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding for continuous flow time t in [0, 1]."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (B,) in [0, 1]
        half_dim = self.embed_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class CoupledGeoActionFlow(nn.Module):
    """
    Joint Vector Field Network v_θ(u_t, t, c) for simultaneous Action + Geometry generation.
    """

    def __init__(
        self,
        cond_dim: int = 1024,
        action_dim: int = 7,
        geo_dim: int = 128,
        horizon: int = 8,
        hidden_dim: int = 512,
        num_layers: int = 4
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.action_dim = action_dim
        self.geo_dim = geo_dim
        self.horizon = horizon
        self.total_dim = action_dim + geo_dim  # 7 + 128 = 135
        self.flat_dim = self.horizon * self.total_dim

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(embed_dim=256),
            nn.Linear(256, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Condition projection
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        # State input projection
        self.state_proj = nn.Linear(self.flat_dim, hidden_dim)

        # Joint Vector Field MLP / Transformer Blocks
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_layers)
        ])

        # Velocity output projection
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.flat_dim)
        )

    def forward_velocity(
        self,
        u_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes velocity field v_θ(u_t, t, c).
        
        Args:
            u_t: (B, H, total_dim) or (B, flat_dim)
            t: (B,) continuous timestep in [0, 1]
            cond: (B, cond_dim) multimodal condition from VLM
            
        Returns:
            v_t: (B, H, total_dim) predicted velocity field
        """
        B = u_t.shape[0]
        if u_t.dim() == 3:
            u_flat = u_t.reshape(B, self.flat_dim)
        else:
            u_flat = u_t

        t_emb = self.time_embed(t)      # (B, hidden_dim)
        c_emb = self.cond_proj(cond)    # (B, hidden_dim)
        s_emb = self.state_proj(u_flat) # (B, hidden_dim)

        h = s_emb + t_emb + c_emb

        for layer in self.layers:
            h = h + layer(h)

        v_flat = self.out_proj(h)
        return v_flat.view(B, self.horizon, self.total_dim)

    def compute_flow_loss(
        self,
        actions_gt: torch.Tensor,
        geo_tracks_gt: torch.Tensor,
        cond: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computes single unified Optimal Transport Flow-Matching Loss.
        
        Args:
            actions_gt: (B, H, action_dim) ground truth robot actions
            geo_tracks_gt: (B, H, geo_dim) ground truth 3D point-track displacements
            cond: (B, cond_dim) VLM conditioning embedding
        """
        B = actions_gt.shape[0]
        device = actions_gt.device

        # Form unified target state u_1 = [a, Δp]
        u_1 = torch.cat([actions_gt, geo_tracks_gt], dim=-1)  # (B, H, total_dim)

        # Sample standard Gaussian noise u_0
        u_0 = torch.randn_like(u_1)

        # Sample continuous time t in [0, 1]
        t = torch.rand(B, device=device)

        # Linear optimal transport interpolation: u_t = (1 - t) * u_0 + t * u_1
        t_expand = t.view(B, 1, 1)
        u_t = (1.0 - t_expand) * u_0 + t_expand * u_1

        # Target velocity vector field: v_target = u_1 - u_0
        v_target = u_1 - u_0

        # Predict velocity field
        v_pred = self.forward_velocity(u_t, t, cond)

        # Single Joint Flow-Matching MSE Loss
        loss_total = F.mse_loss(v_pred, v_target)

        # Decompose for telemetry monitoring
        v_pred_act, v_pred_geo = v_pred[..., :self.action_dim], v_pred[..., self.action_dim:]
        v_tgt_act, v_tgt_geo = v_target[..., :self.action_dim], v_target[..., self.action_dim:]

        loss_action = F.mse_loss(v_pred_act, v_tgt_act)
        loss_geo = F.mse_loss(v_pred_geo, v_tgt_geo)

        return {
            "loss_coupled_flow": loss_total,
            "loss_action_component": loss_action,
            "loss_geo_component": loss_geo
        }

    @torch.no_grad()
    def sample_trajectory(
        self,
        cond: torch.Tensor,
        num_steps: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrates the ODE du/dt = v_θ(u_t, t, c) from t=0 to 1 via Euler integration.
        
        Returns:
            predicted_actions: (B, H, action_dim)
            predicted_geo_tracks: (B, H, geo_dim)
        """
        B = cond.shape[0]
        device = cond.device

        # Start from pure noise at t=0
        u = torch.randn(B, self.horizon, self.total_dim, device=device)
        dt = 1.0 / num_steps

        for step_idx in range(num_steps):
            t_val = step_idx * dt
            t = torch.full((B,), t_val, device=device, dtype=torch.float32)
            v = self.forward_velocity(u, t, cond)
            u = u + v * dt

        # Split joint state u into Action and 3D Geometry
        pred_actions = u[..., :self.action_dim]
        pred_geo_tracks = u[..., self.action_dim:]

        return pred_actions, pred_geo_tracks
