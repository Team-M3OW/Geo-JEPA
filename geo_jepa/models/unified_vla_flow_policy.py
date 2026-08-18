"""
Geo-JEPA: Unified Multimodal Vision-Language-Action (VLA) Policy with Coupled 3D Flow.

Architecture:
1. Multi-Modal Vision-Language Transformer:
   - Vision Tokens: Visual Feature Extractor (RGB -> Z_vis)
   - Language Tokens: Qwen2.5 / Transformer Text Encoder (Prompt -> Z_lang)
   - Cross-Attention: Z_multimodal = CrossAttention(Z_vis, Z_lang)
2. 3D Geometric Ray Head:
   - Predicts 3D line-of-sight rays & metric contact basins from multimodal tokens.
3. Coupled Geometric Flow Matching Head:
   - Generates 8-step continuous motor trajectory chunks [a_1, ..., a_8] in R^(8 x 7).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from geo_jepa.models.coupled_geo_action_flow import CoupledGeoActionFlow


class MultimodalCrossAttentionBlock(nn.Module):
    """Cross-Attention layer fusing visual patch tokens with language prompt tokens."""
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x_vis: torch.Tensor, x_lang: torch.Tensor) -> torch.Tensor:
        # x_vis: [B, N_vis, D], x_lang: [B, N_lang, D]
        attn_out, _ = self.cross_attn(query=self.norm1(x_vis), key=x_lang, value=x_lang)
        x = x_vis + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class UnifiedVLAFlowPolicy(nn.Module):
    """
    True Vision-Language-Action Policy conditioned natively on task text.
    """
    def __init__(
        self,
        embed_dim: int = 512,
        action_dim: int = 7,
        horizon: int = 8,
        vocab_size: int = 151936  # Qwen2.5 vocab size
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.action_dim = action_dim

        # 1. Vision Tokenizer (Patch Conv -> [B, 16, embed_dim])
        self.vis_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4))  # 16 visual tokens
        )

        # 2. Text Token Embedding (Qwen Tokenizer IDs -> [B, L, embed_dim])
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        # 3. Proprioception Projector (EEF [3] + Gripper [2] -> [B, 1, embed_dim])
        self.proprio_proj = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        # 4. Multimodal Cross-Attention Fusion
        self.cross_modal = MultimodalCrossAttentionBlock(embed_dim=embed_dim, num_heads=8)

        # 5. 3D Action Ray Head (predicts 3D line-of-sight & target basins)
        self.ray_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 6)  # [target_dx, target_dy, target_dz, rec_dx, rec_dy, rec_dz]
        )

        # 6. Coupled Geometric-Action Flow Matching Head
        self.flow_head = CoupledGeoActionFlow(
            cond_dim=embed_dim,
            action_dim=action_dim,
            geo_dim=128,
            horizon=horizon,
            hidden_dim=512,
            num_layers=4
        )

        # Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        except Exception:
            self.tokenizer = None

    def encode_text(self, text_prompts: List[str], device: torch.device) -> torch.Tensor:
        """Tokenizes and embeds text prompts."""
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                text_prompts,
                padding=True,
                truncation=True,
                max_length=32,
                return_tensors="pt"
            ).to(device)
            input_ids = tokens["input_ids"]
        else:
            # Fallback deterministic ASCII hashing for offline testing
            input_ids = torch.zeros((len(text_prompts), 16), dtype=torch.long, device=device)
            for b, p in enumerate(text_prompts):
                for i, char in enumerate(p[:16]):
                    input_ids[b, i] = ord(char) % 1000

        x_lang = self.text_embed(input_ids)
        return self.text_proj(x_lang)

    def forward_multimodal(
        self,
        rgb_image: torch.Tensor,
        task_prompts: List[str],
        eef_pos: torch.Tensor,
        gripper_q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multimodal VLA forward pass.
        Returns:
          pred_action_chunks: [B, horizon, 7]
          pred_3d_rays: [B, 6]
        """
        device = rgb_image.device
        B = rgb_image.shape[0]

        # 1. Vision Tokens: [B, 16, embed_dim]
        vis_feat = self.vis_encoder(rgb_image)  # [B, embed_dim, 4, 4]
        x_vis = vis_feat.flatten(2).permute(0, 2, 1)  # [B, 16, embed_dim]

        # 2. Text Tokens: [B, L, embed_dim]
        x_lang = self.encode_text(task_prompts, device)

        # 3. Proprioception Token: [B, 1, embed_dim]
        proprio = torch.cat([eef_pos, gripper_q], dim=-1)
        x_proprio = self.proprio_proj(proprio).unsqueeze(1)

        # Combine Vision + Proprioception
        x_vis_full = torch.cat([x_vis, x_proprio], dim=1)  # [B, 17, embed_dim]

        # 4. Cross-Modal Fusion
        z_fused = self.cross_modal(x_vis_full, x_lang)  # [B, 17, embed_dim]
        z_cond = z_fused.mean(dim=1)  # [B, embed_dim] Global multimodal condition

        # 5. Predict 3D Spatial Action Rays
        pred_rays = self.ray_head(z_cond)

        # 6. Sample 8-step Continuous Action Trajectory via Flow Matching
        pred_act, pred_geo = self.flow_head.sample_trajectory(z_cond, num_steps=4)

        return pred_act, pred_rays
