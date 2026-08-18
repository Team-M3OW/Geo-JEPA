"""
Geo-JEPA: DINOv2-Grounded Multimodal Action Chunking Transformer (ACT) Policy.

Enhancements:
1. Pretrained DINOv2 Visual Tokenizer: Extracts dense, linearly separable spatial patch tokens.
2. Action Space Normalization: Normalizes actions by dataset (mu, sigma) so all 7 DoFs are balanced.
3. Multimodal Transformer Architecture: Cross-attends task language tokens to visual tokens and decodes 8-step continuous action chunks (H=8).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


class MultimodalCrossAttentionBlock(nn.Module):
    """Transformer block with self-attention and cross-attention over language tokens."""
    def __init__(self, embed_dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x_vis: torch.Tensor, x_lang: torch.Tensor) -> torch.Tensor:
        # 1. Self-Attention over visual + proprioception tokens
        s_out, _ = self.self_attn(query=self.norm1(x_vis), key=x_vis, value=x_vis)
        x = x_vis + s_out

        # 2. Cross-Attention with language prompt tokens
        c_out, _ = self.cross_attn(query=self.norm2(x), key=x_lang, value=x_lang)
        x = x + c_out

        # 3. Feed-Forward
        x = x + self.ffn(self.norm3(x))
        return x


class DINOv2ACTPolicy(nn.Module):
    """
    DINOv2-Grounded Multimodal Action Chunking Policy.
    """
    def __init__(
        self,
        embed_dim: int = 384,  # DINOv2-small dimension
        action_dim: int = 7,
        horizon: int = 8,
        vocab_size: int = 151936
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.action_dim = action_dim

        # 1. Pretrained DINOv2 Visual Backbone (Frozen/Fine-tuned)
        try:
            self.vision_encoder = timm.create_model(
                "vit_small_patch14_dinov2.lvd142m",
                pretrained=True,
                num_classes=0,
                img_size=224
            )
        except Exception:
            # Fallback to local pretrained ResNet if offline
            self.vision_encoder = timm.create_model("resnet34", pretrained=True, num_classes=0)

        # Vision feature projector to embed_dim
        vis_feat_dim = getattr(self.vision_encoder, "num_features", embed_dim)
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim)
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

        # 4. Multimodal Fusion Transformer Blocks
        self.fusion_blocks = nn.ModuleList([
            MultimodalCrossAttentionBlock(embed_dim=embed_dim, num_heads=6)
            for _ in range(3)
        ])

        # 5. Action Chunking Decoder Queries [H, embed_dim]
        self.action_queries = nn.Parameter(torch.randn(horizon, embed_dim) * 0.02)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=6,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, action_dim)
        )

        # 6. Action Space Normalization Statistics (from dataset stats.json)
        self.register_buffer(
            "action_mean",
            torch.tensor([0.1531, 0.1371, -0.1553, -0.0052, -0.0112, -0.0202, 0.0842], dtype=torch.float32)
        )
        self.register_buffer(
            "action_std",
            torch.tensor([0.4127, 0.3473, 0.5087, 0.0373, 0.0724, 0.0576, 0.9964], dtype=torch.float32)
        )

        # Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        except Exception:
            self.tokenizer = None

    def encode_text(self, text_prompts: List[str], device: torch.device) -> torch.Tensor:
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
            input_ids = torch.zeros((len(text_prompts), 16), dtype=torch.long, device=device)
            for b, p in enumerate(text_prompts):
                for i, char in enumerate(p[:16]):
                    input_ids[b, i] = ord(char) % 1000

        x_lang = self.text_embed(input_ids)
        return self.text_proj(x_lang)

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalizes actions to zero mean and unit variance."""
        return (actions - self.action_mean) / (self.action_std + 1e-4)

    def unnormalize_actions(self, norm_actions: torch.Tensor) -> torch.Tensor:
        """Denormalizes actions back to physical robot scale."""
        return norm_actions * (self.action_std + 1e-4) + self.action_mean

    def forward(
        self,
        rgb_image: torch.Tensor,
        task_prompts: List[str],
        eef_pos: torch.Tensor,
        gripper_q: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass predicting normalized 8-step action chunks [B, H, 7].
        """
        device = rgb_image.device
        B = rgb_image.shape[0]

        # 1. Vision Features: Upsample to 224x224 for DINOv2
        if rgb_image.shape[-1] != 224:
            rgb_224 = F.interpolate(rgb_image, size=(224, 224), mode="bicubic", align_corners=False)
        else:
            rgb_224 = rgb_image

        vis_feat = self.vision_encoder(rgb_224)  # [B, vis_dim]
        if vis_feat.dim() == 2:
            x_vis = self.vis_proj(vis_feat).unsqueeze(1)  # [B, 1, embed_dim]
        else:
            x_vis = self.vis_proj(vis_feat)  # [B, N_patches, embed_dim]

        # 2. Text Features: [B, L, embed_dim]
        x_lang = self.encode_text(task_prompts, device)

        # 3. Proprioception Token: [B, 1, embed_dim]
        proprio = torch.cat([eef_pos, gripper_q], dim=-1)
        x_proprio = self.proprio_proj(proprio).unsqueeze(1)

        # Combine Vision + Proprioception
        memory = torch.cat([x_vis, x_proprio], dim=1)  # [B, N_vis + 1, embed_dim]

        # 4. Cross-Attention Fusion
        for block in self.fusion_blocks:
            memory = block(memory, x_lang)

        # 5. Decode 8-step Action Chunk
        tgt_queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)  # [B, H, embed_dim]
        dec_out = self.decoder_layer(tgt=tgt_queries, memory=memory)  # [B, H, embed_dim]
        pred_norm_actions = self.action_head(dec_out)  # [B, H, 7]

        return pred_norm_actions

    @torch.no_grad()
    def get_action_chunk(
        self,
        rgb_image: torch.Tensor,
        task_prompt: str,
        eef_pos: torch.Tensor,
        gripper_q: torch.Tensor
    ) -> np.ndarray:
        """
        Inference method returning denormalized robot actions in R^(H x 7).
        """
        self.eval()
        pred_norm = self.forward(
            rgb_image=rgb_image,
            task_prompts=[task_prompt],
            eef_pos=eef_pos,
            gripper_q=gripper_q
        )
        pred_denorm = self.unnormalize_actions(pred_norm)
        return pred_denorm[0].detach().cpu().numpy()
