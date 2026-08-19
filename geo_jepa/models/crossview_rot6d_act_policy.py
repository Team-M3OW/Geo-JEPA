"""
Geo-JEPA: Cross-View Attention Bridge + 6D Continuous Rotation Multimodal ACT Policy.

Key Architectural Foundations:
1. Architectural Fix 3: Cross-View Attention Bridge (Global AgentView ◄► Wrist Camera)
   - 2D Sinusoidal Spatial Coordinate Grid Embeddings.
   - Bidirectional Cross-Attention between global workspace context and egocentric wrist context.
2. Architectural Fix 4: 6D Continuous Rotation Representation (Zhou et al., CVPR 2019)
   - Predicts continuous 6D rotation vectors [a1, a2] in R^6.
   - Gram-Schmidt orthogonalization guarantees non-singular, continuous SO(3) rotation matrices with smooth gradients.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from transformers import AutoTokenizer


def compute_rotation_matrix_from_ortho6d(ortho6d: torch.Tensor) -> torch.Tensor:
    """
    Gram-Schmidt orthogonalization from 6D representation to SO(3) 3x3 matrix.
    Args:
        ortho6d: (..., 6)
    Returns:
        rot_matrix: (..., 3, 3)
    """
    x_raw = ortho6d[..., 0:3]  # (..., 3)
    y_raw = ortho6d[..., 3:6]  # (..., 3)

    x = F.normalize(x_raw, p=2, dim=-1, eps=1e-6)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, p=2, dim=-1, eps=1e-6)
    y = torch.cross(z, x, dim=-1)

    matrix = torch.stack((x, y, z), dim=-1)  # (..., 3, 3)
    return matrix


def matrix_to_euler_angles(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts 3x3 rotation matrix to continuous Euler angles (roll, pitch, yaw) in radians.
    """
    # R31, R32, R33, R21, R11
    r11 = matrix[..., 0, 0]
    r21 = matrix[..., 1, 0]
    r31 = matrix[..., 2, 0]
    r32 = matrix[..., 2, 1]
    r33 = matrix[..., 2, 2]

    pitch = torch.atan2(-r31, torch.sqrt(r32**2 + r33**2 + 1e-8))
    roll = torch.atan2(r32, r33)
    yaw = torch.atan2(r21, r11)

    return torch.stack([roll, pitch, yaw], dim=-1)


def euler_to_rot6d(euler_angles: torch.Tensor) -> torch.Tensor:
    """
    Converts Euler angles (B, ..., 3) [roll, pitch, yaw] to 6D continuous representation.
    """
    roll = euler_angles[..., 0]
    pitch = euler_angles[..., 1]
    yaw = euler_angles[..., 2]

    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    # First two columns of R = Rz * Ry * Rx
    r11 = cy * cp
    r21 = sy * cp
    r31 = -sp

    r12 = cy * sp * sr - sy * cr
    r22 = sy * sp * sr + cy * cr
    r32 = cp * sr

    return torch.stack([r11, r21, r31, r12, r22, r32], dim=-1)


class SpatialCoordinate2DGrid(nn.Module):
    """Generates normalized 2D (x, y) spatial grid embeddings for visual patch tokens."""
    def __init__(self, embed_dim: int = 384, grid_size: int = 16):
        super().__init__()
        self.grid_size = grid_size
        
        # Linear projector from normalized [x, y] in [-1, 1] to embed_dim
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # Precompute grid
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, grid_size),
            torch.linspace(-1.0, 1.0, grid_size),
            indexing="ij"
        )
        grid = torch.stack([x.flatten(), y.flatten()], dim=-1)  # [grid_size^2, 2]
        self.register_buffer("spatial_grid", grid)

    def forward(self, x_patches: torch.Tensor) -> torch.Tensor:
        # x_patches: [B, N_patches, embed_dim]
        N = x_patches.shape[1]
        grid = self.spatial_grid[:N].unsqueeze(0)  # [1, N, 2]
        pos_emb = self.coord_mlp(grid)  # [1, N, embed_dim]
        return x_patches + pos_emb


class CrossViewAttentionBridge(nn.Module):
    """
    Bidirectional Cross-Attention Bridge between Global AgentView and Egocentric Wrist Camera.
    """
    def __init__(self, embed_dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.agent_to_wrist = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.wrist_to_agent = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        self.norm_agent1 = nn.LayerNorm(embed_dim)
        self.norm_wrist1 = nn.LayerNorm(embed_dim)
        self.norm_agent2 = nn.LayerNorm(embed_dim)
        self.norm_wrist2 = nn.LayerNorm(embed_dim)

        self.ffn_agent = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.ffn_wrist = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x_agent: torch.Tensor, x_wrist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Wrist attends to Global AgentView (Local coordinates grounded in global space)
        w_out, _ = self.agent_to_wrist(
            query=self.norm_wrist1(x_wrist),
            key=self.norm_agent1(x_agent),
            value=x_agent
        )
        x_wrist = x_wrist + w_out
        x_wrist = x_wrist + self.ffn_wrist(self.norm_wrist2(x_wrist))

        # 2. AgentView attends to Wrist (Global camera gains high-resolution contact focus)
        a_out, _ = self.wrist_to_agent(
            query=self.norm_agent1(x_agent),
            key=self.norm_wrist1(x_wrist),
            value=x_wrist
        )
        x_agent = x_agent + a_out
        x_agent = x_agent + self.ffn_agent(self.norm_agent2(x_agent))

        return x_agent, x_wrist


class CrossViewRot6dACTPolicy(nn.Module):
    """
    SOTA Multi-Task ACT Policy with:
    - Cross-View Attention Bridge (Global ◄► Wrist) with 2D Spatial Grids.
    - 6D Continuous Rotation Action Head.
    - Qwen2.5 Language Grounding & Action Chunking (H=16).
    """
    def __init__(
        self,
        embed_dim: int = 384,
        horizon: int = 16,
        vocab_size: int = 151936
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.horizon = horizon

        # 1. Pretrained DINOv2 Visual Tokenizer
        try:
            self.vision_encoder = timm.create_model(
                "vit_small_patch14_dinov2.lvd142m",
                pretrained=True,
                num_classes=0,
                img_size=224
            )
        except Exception:
            self.vision_encoder = timm.create_model("resnet34", pretrained=True, num_classes=0)

        vis_dim = getattr(self.vision_encoder, "num_features", embed_dim)
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 2. Spatial 2D Coordinate Grid Positional Encodings
        self.spatial_grid = SpatialCoordinate2DGrid(embed_dim=embed_dim, grid_size=16)

        # 3. Cross-View Attention Bridge (Fix 3)
        self.cross_view_bridge = CrossViewAttentionBridge(embed_dim=embed_dim, num_heads=6)

        # 4. Text Tokenizer Embedding
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        # 5. Proprioception Projector
        self.proprio_proj = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        # 6. Multimodal Language Cross-Attention Stack
        self.lang_cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=6, batch_first=True)
        self.lang_norm1 = nn.LayerNorm(embed_dim)
        self.lang_norm2 = nn.LayerNorm(embed_dim)
        self.lang_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # 7. Action Chunk Decoder Queries (H=16)
        self.action_queries = nn.Parameter(torch.randn(horizon, embed_dim) * 0.02)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=6,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )

        # 8. Decoupled Continuous Heads (Fix 4: 6D Rotation)
        self.translation_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3)  # Continuous Translation (dx, dy, dz)
        )
        self.rotation_6d_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 6)  # 6D Continuous Rotation (r1_x, r1_y, r1_z, r2_x, r2_y, r2_z)
        )
        self.gripper_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)  # Gripper open/close logit
        )

        # 9. Translation Normalization Statistics
        self.register_buffer(
            "trans_mean",
            torch.tensor([0.1531, 0.1371, -0.1553], dtype=torch.float32)
        )
        self.register_buffer(
            "trans_std",
            torch.tensor([0.4127, 0.3473, 0.5087], dtype=torch.float32)
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

    def forward(
        self,
        agentview_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor,
        task_prompts: List[str],
        eef_pos: torch.Tensor,
        gripper_q: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        device = agentview_rgb.device
        B = agentview_rgb.shape[0]

        # 1. Extract Visual Features
        rgb_agent_224 = F.interpolate(agentview_rgb, size=(224, 224), mode="bicubic", align_corners=False)
        rgb_wrist_224 = F.interpolate(wrist_rgb, size=(224, 224), mode="bicubic", align_corners=False)

        f_agent = self.vision_encoder(rgb_agent_224)
        f_wrist = self.vision_encoder(rgb_wrist_224)

        x_agent = self.vis_proj(f_agent).unsqueeze(1) if f_agent.dim() == 2 else self.vis_proj(f_agent)
        x_wrist = self.vis_proj(f_wrist).unsqueeze(1) if f_wrist.dim() == 2 else self.vis_proj(f_wrist)

        # 2. Add 2D Spatial Coordinate Grid Embeddings (Fix 3)
        x_agent = self.spatial_grid(x_agent)
        x_wrist = self.spatial_grid(x_wrist)

        # 3. Cross-View Attention Bridge (Fix 3)
        x_agent, x_wrist = self.cross_view_bridge(x_agent, x_wrist)

        # 4. Language & Proprioception Encoding
        x_lang = self.encode_text(task_prompts, device)
        proprio = torch.cat([eef_pos, gripper_q], dim=-1)
        x_proprio = self.proprio_proj(proprio).unsqueeze(1)

        # 5. Full Multimodal Fusion Memory
        vis_memory = torch.cat([x_agent, x_wrist, x_proprio], dim=1)
        l_out, _ = self.lang_cross_attn(
            query=self.lang_norm1(vis_memory),
            key=x_lang,
            value=x_lang
        )
        memory = vis_memory + l_out
        memory = memory + self.lang_ffn(self.lang_norm2(memory))

        # 6. Action Chunk Decoder (H=16)
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        dec_out = self.decoder_layer(tgt=queries, memory=memory)  # [B, 16, embed_dim]

        # 7. Decoupled Heads: Translation (Norm) + 6D Continuous Rotation + Gripper (Fix 4)
        pred_trans_norm = self.translation_head(dec_out)  # [B, 16, 3]
        pred_rot6d = self.rotation_6d_head(dec_out)        # [B, 16, 6]
        pred_gripper = self.gripper_head(dec_out)          # [B, 16, 1]

        return {
            "pred_trans_norm": pred_trans_norm,
            "pred_rot6d": pred_rot6d,
            "pred_gripper": pred_gripper
        }

    @torch.no_grad()
    def get_action_chunk(
        self,
        agentview_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor,
        task_prompt: str,
        eef_pos: torch.Tensor,
        gripper_q: torch.Tensor
    ) -> np.ndarray:
        self.eval()
        outputs = self.forward(
            agentview_rgb=agentview_rgb,
            wrist_rgb=wrist_rgb,
            task_prompts=[task_prompt],
            eef_pos=eef_pos,
            gripper_q=gripper_q
        )

        # Denormalize translation
        pred_trans = outputs["pred_trans_norm"] * (self.trans_std + 1e-4) + self.trans_mean  # [1, 16, 3]

        # Convert 6D rotation to continuous Euler angles (Fix 4)
        rot_matrix = compute_rotation_matrix_from_ortho6d(outputs["pred_rot6d"])             # [1, 16, 3, 3]
        pred_euler = matrix_to_euler_angles(rot_matrix)                                       # [1, 16, 3]

        # Gripper thresholding
        pred_grp = torch.where(outputs["pred_gripper"] > 0.0, torch.tensor(1.0, device=pred_trans.device), torch.tensor(-1.0, device=pred_trans.device))

        # Full 7-DoF Action Chunk: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        full_actions = torch.cat([pred_trans, pred_euler, pred_grp], dim=-1)  # [1, 16, 7]
        return full_actions[0].detach().cpu().numpy()
