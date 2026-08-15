"""
Dual-Head World Model Predictor for Geo-JEPA.

Extends the action-conditioned Vision Transformer Predictor to concurrently predict:
1. Future semantic latent states (V-JEPA2 targets)
2. Future geometric dynamic states (canonicalized VGGT 3D points / point-track displacements)

Guarantees leakage-free operation: targets are constructed via frozen encoders with stop-gradient.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, "/home/kavinder/geo-jepa-dev/VLA-JEPA")
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC


class DualHeadVisionTransformerPredictor(nn.Module):
    """
    Action-conditioned world model with dual prediction heads:
    - Semantic Head: projects to V-JEPA2 latent space
    - Geometric Head: projects to canonicalized VGGT geometric state (track displacements or point maps)
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (256, 256),
        patch_size: int = 16,
        num_frames: int = 4,
        tubelet_size: int = 1,
        embed_dim_semantic: int = 2048,   # V-JEPA2 multi-view feature dimension
        geo_target_dim: int = 128,         # Dimension of geometric target (e.g. N_tracks * 2 or patch_dim)
        predictor_embed_dim: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        action_embed_dim: int = 2048,      # Qwen-VL hidden size
        num_add_tokens: int = 8,
        geo_target_type: str = "track_displacement",  # 'track_displacement' or 'point_map'
        **kwargs
    ) -> None:
        super().__init__()
        self.geo_target_type = geo_target_type
        self.embed_dim_semantic = embed_dim_semantic
        self.geo_target_dim = geo_target_dim
        self.predictor_embed_dim = predictor_embed_dim

        # Base action-conditioned transformer trunk
        self.base_predictor = VisionTransformerPredictorAC(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=embed_dim_semantic,
            predictor_embed_dim=predictor_embed_dim,
            depth=depth,
            num_heads=num_heads,
            action_embed_dim=action_embed_dim,
            num_add_tokens=num_add_tokens,
            **kwargs
        )

        # Head 1: Semantic Future State Prediction (V-JEPA2)
        # We reuse base_predictor's predictor_proj & norm for semantic output
        self.semantic_head = self.base_predictor.predictor_proj

        # Head 2: Geometric Future State Prediction (VGGT canonical dynamics)
        self.geo_norm = nn.LayerNorm(predictor_embed_dim)
        self.geo_head = nn.Sequential(
            nn.Linear(predictor_embed_dim, predictor_embed_dim, bias=True),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, geo_target_dim, bias=True)
        )

        self._init_geo_head()

    def _init_geo_head(self):
        for m in self.geo_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward_trunk(
        self,
        context_states: torch.Tensor,
        action_tokens: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer trunk before task projection heads.
        
        Args:
            context_states: [B, (T-1) * num_patches, embed_dim_semantic]
            action_tokens: [B, T * num_action_tokens, action_embed_dim]
            
        Returns:
            Trunk hidden features of shape [B, (T-1) * num_patches, predictor_embed_dim]
        """
        x = self.base_predictor.predictor_embed(context_states)
        B, N_ctxt, D = x.size()
        T = N_ctxt // (self.base_predictor.grid_height * self.base_predictor.grid_width)

        # Interleave action tokens
        a = self.base_predictor.action_encoder(action_tokens)
        a = a.view(B, T, -1, D)
        cond_tokens = a.shape[2]
        x = x.view(B, T, self.base_predictor.grid_height * self.base_predictor.grid_width, D)

        if self.base_predictor.use_extrinsics and extrinsics is not None:
            cond_tokens += 1
            e = self.base_predictor.extrinsics_encoder(extrinsics).unsqueeze(2)
            x = torch.cat([a, e, x], dim=2).flatten(1, 2)
        else:
            x = torch.cat([a, x], dim=2).flatten(1, 2)

        from starVLA.model.modules.world_model.vj2_modules import build_action_block_causal_attention_mask
        attn_mask = build_action_block_causal_attention_mask(
            T, self.base_predictor.grid_height, self.base_predictor.grid_width, add_tokens=cond_tokens
        ).to(x.device, non_blocking=True)

        for blk in self.base_predictor.predictor_blocks:
            x = blk(
                x,
                mask=None,
                attn_mask=attn_mask,
                T=T,
                H=self.base_predictor.grid_height,
                W=self.base_predictor.grid_width,
                action_tokens=cond_tokens,
            )

        # Squeeze out condition tokens
        x = x.view(B, T, cond_tokens + self.base_predictor.grid_height * self.base_predictor.grid_width, D)
        x = x[:, :, cond_tokens:, :].flatten(1, 2)
        return x

    def forward(
        self,
        context_states: torch.Tensor,
        action_tokens: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing both semantic and geometric future state predictions.
        
        Returns:
            predicted_sem_states: [B, (T-1) * num_patches, embed_dim_semantic]
            predicted_geo_states: [B, (T-1) * num_patches, geo_target_dim] or [B, (T-1), geo_target_dim]
        """
        trunk_features = self.forward_trunk(context_states, action_tokens, extrinsics=extrinsics)

        # Semantic projection
        sem_feat = self.base_predictor.predictor_norm(trunk_features)
        pred_sem_states = self.semantic_head(sem_feat)

        # Geometric projection
        geo_feat = self.geo_norm(trunk_features)
        pred_geo_states = self.geo_head(geo_feat)

        return pred_sem_states, pred_geo_states

    def compute_dual_wm_loss(
        self,
        pred_sem_states: torch.Tensor,
        gt_sem_states: torch.Tensor,
        pred_geo_states: torch.Tensor,
        gt_geo_states: torch.Tensor,
        gamma: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute leakage-free semantic and geometric world model prediction losses.
        
        Args:
            pred_sem_states: Predicted semantic latent states
            gt_sem_states: Target V-JEPA2 latent states (detached)
            pred_geo_states: Predicted geometric dynamic states
            gt_geo_states: Target VGGT geometric dynamic states (detached)
            gamma: Independent geometric world model loss multiplier
            
        Returns:
            Dict containing 'wm_loss_sem', 'wm_loss_geo', 'wm_loss_total'
        """
        # Semantic L1 loss (VLA-JEPA formulation)
        loss_sem = F.l1_loss(pred_sem_states, gt_sem_states.detach(), reduction="mean")

        # Geometric L1 / Huber loss
        # If geometric targets are per-timestep summary vectors (e.g. track displacements)
        if pred_geo_states.shape != gt_geo_states.shape:
            # Pool spatial dimensions across patches if needed
            if pred_geo_states.dim() == 3 and gt_geo_states.dim() == 3:
                if pred_geo_states.shape[1] != gt_geo_states.shape[1]:
                    B, N_patches, D_g = pred_geo_states.shape
                    T_minus_1 = gt_geo_states.shape[1]
                    pred_geo_pooled = pred_geo_states.view(B, T_minus_1, -1, D_g).mean(dim=2)
                    loss_geo = F.smooth_l1_loss(pred_geo_pooled, gt_geo_states.detach(), reduction="mean")
                else:
                    loss_geo = F.smooth_l1_loss(pred_geo_states, gt_geo_states.detach(), reduction="mean")
            else:
                loss_geo = F.smooth_l1_loss(pred_geo_states, gt_geo_states.detach(), reduction="mean")
        else:
            loss_geo = F.smooth_l1_loss(pred_geo_states, gt_geo_states.detach(), reduction="mean")

        loss_total = loss_sem + (gamma * loss_geo)

        return {
            "wm_loss_sem": loss_sem,
            "wm_loss_geo": loss_geo,
            "wm_loss_total": loss_total,
        }
