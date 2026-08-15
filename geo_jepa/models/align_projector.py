"""
Spatial-Forcing Geometric Alignment Projector & Loss Module.

Direct port from official Spatial-Forcing (openvla-SF/prismatic/models/projectors.py & pooling_utils.py):
1. 2-layer MLP projection mapping VLM mid-layer visual tokens to VGGT feature space (2 * D_vggt = 2048).
2. UV Positional Embedding addition to VGGT target features to preserve token-position correspondence.
3. Bilinear feature interpolation to match spatial token dimensions between VLM and VGGT.
4. Cosine similarity alignment loss: L_geo = 1 - mean(cosine_sim(vlm_proj, vggt_target)).
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Create a 2D normalized UV coordinate grid in range [-1, 1].
    """
    u = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
    v = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    grid = torch.stack([u_grid * aspect_ratio, v_grid], dim=-1)  # (H, W, 2)
    return grid


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100.0) -> torch.Tensor:
    """
    Generate 1D positional embedding using sine and cosine functions.
    """
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=device)
    omega /= (embed_dim / 2.0)
    omega = 1.0 / (omega_0 ** omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100.0) -> torch.Tensor:
    """
    Convert 2D position grid (H, W, 2) to sinusoidal embeddings (H, W, embed_dim).
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # (H*W, 2)

    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)  # (H*W, D/2)
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)  # (H*W, D/2)

    emb = torch.cat([emb_x, emb_y], dim=-1)  # (H*W, D)
    return emb.view(H, W, embed_dim)


def apply_pos_embed(
    x: torch.Tensor,
    img_w: int,
    img_h: int,
    ratio: float = 0.1
) -> torch.Tensor:
    """
    Apply 2D UV positional embedding to spatial feature maps x: (B*N, D, H_p, W_p).
    """
    B_N, D, patch_h, patch_w = x.shape
    pos_grid = create_uv_grid(patch_w, patch_h, aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
    pos_embed = position_grid_to_embed(pos_grid, D)  # (H_p, W_p, D)
    pos_embed = pos_embed.permute(2, 0, 1).unsqueeze(0)  # (1, D, H_p, W_p)
    return x + (pos_embed * ratio)


def interpolate_pooling(
    hidden: torch.Tensor,
    patch_hw: Tuple[int, int],
    img_hw: Tuple[int, int],
    target_num_tokens: int,
    mode: str = "bilinear",
    use_vggt_pe: bool = True,
    pe_ratio: float = 0.1
) -> torch.Tensor:
    """
    Interpolate VGGT hidden features to match the resolution of VLM visual tokens.
    
    Args:
        hidden: VGGT spatial features of shape (B, N_views, num_patches, D)
        patch_hw: (patch_h, patch_w) of VGGT feature grid (e.g. 37, 37)
        img_hw: (H, W) original image size (e.g. 518, 518)
        target_num_tokens: Number of visual tokens in VLM representation per view
        mode: Interpolation mode ('bilinear')
        use_vggt_pe: Whether to add UV positional embeddings before pooling
        pe_ratio: Weight ratio for positional embedding (default: 0.1 from paper)
        
    Returns:
        Pooled VGGT features of shape (B, N_views * target_num_tokens, D) or (B, target_num_tokens, D)
    """
    bs, N, S, D = hidden.shape
    patch_h, patch_w = patch_hw
    img_h, img_w = img_hw
    
    # Calculate target spatial grid
    target_tokens_per_view = target_num_tokens // N if (target_num_tokens % N == 0) else target_num_tokens
    target_side = int(np.round(np.sqrt(target_tokens_per_view)))
    
    # Reshape to (B*N, D, patch_h, patch_w)
    hidden_2d = hidden.permute(0, 1, 3, 2).reshape(bs * N, D, patch_h, patch_w)
    
    if use_vggt_pe:
        hidden_2d = apply_pos_embed(hidden_2d, img_w, img_h, ratio=pe_ratio)
        
    # Bilinear interpolate to target grid
    if (patch_h, patch_w) != (target_side, target_side):
        hidden_pooled = F.interpolate(
            hidden_2d,
            size=(target_side, target_side),
            mode=mode,
            align_corners=True
        )
    else:
        hidden_pooled = hidden_2d
        
    # Reshape back to (B, N_views * target_tokens_per_view, D)
    hidden_out = hidden_pooled.reshape(bs, N, D, -1).permute(0, 1, 3, 2).reshape(bs, -1, D)
    
    # Exact token count match guarantee
    if hidden_out.shape[1] != target_num_tokens:
        hidden_out = F.interpolate(
            hidden_out.permute(0, 2, 1),
            size=target_num_tokens,
            mode="linear",
            align_corners=True
        ).permute(0, 2, 1)
        
    return hidden_out


class AlignProjector(nn.Module):
    """
    Projects intermediate VLM visual representations and computes cosine similarity
    alignment with frozen VGGT backbone features.
    
    Follows Spatial-Forcing specification (Linear -> GELU -> Linear + optional LayerNorm).
    """

    def __init__(
        self,
        vlm_dim: int = 2048,
        vggt_dim: int = 1024,
        align_loss_type: str = "cosine",
        use_vlm_norm: bool = False,
    ) -> None:
        super().__init__()
        self.vlm_dim = vlm_dim
        self.vggt_dim = vggt_dim
        self.align_loss_type = align_loss_type
        
        # In VGGT, intermediate concatenated frame+global tokens have dim 2 * embed_dim = 2048
        target_proj_dim = 2 * vggt_dim if vggt_dim != 2048 else vggt_dim
        
        self.vlm_norm = nn.LayerNorm(vlm_dim) if use_vlm_norm else None
        self.fc1 = nn.Linear(vlm_dim, target_proj_dim, bias=True)
        self.act_fn = nn.GELU()
        self.fc2 = nn.Linear(target_proj_dim, target_proj_dim, bias=True)
        
        self.initialize_weights()

    def initialize_weights(self):
        for m in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def project_vlm(self, vlm_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project VLM visual tokens to VGGT feature space.
        Args:
            vlm_embeddings: (B, N_tokens, vlm_dim)
        Returns:
            (B, N_tokens, target_dim)
        """
        if self.vlm_norm is not None:
            vlm_embeddings = self.vlm_norm(vlm_embeddings)
        h = self.fc1(vlm_embeddings)
        h = self.act_fn(h)
        h = self.fc2(h)
        return h

    def compute_align_loss_cosine(
        self,
        vlm_proj: torch.Tensor,
        vggt_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute mean cosine similarity loss: 1 - cosine_similarity(vlm, vggt).
        
        Args:
            vlm_proj: Projected VLM visual tokens (B, N, D)
            vggt_target: Target VGGT features (B, N, D) (detached / stop-gradient)
            mask: Optional boolean mask (B, N) for valid non-padding tokens
        """
        vlm_normed = F.normalize(vlm_proj, p=2, dim=-1)
        vggt_normed = F.normalize(vggt_target.detach(), p=2, dim=-1)
        
        # Cosine similarity per token: sum(vlm * vggt, dim=-1)
        cos_sim = (vlm_normed * vggt_normed).sum(dim=-1)  # (B, N)
        token_loss = 1.0 - cos_sim  # (B, N)
        
        if mask is not None:
            mask_float = mask.to(token_loss.dtype)
            align_loss = (token_loss * mask_float).sum() / torch.clamp(mask_float.sum(), min=1.0)
        else:
            align_loss = token_loss.mean()
            
        return align_loss

    def forward(
        self,
        vlm_visual_tokens: torch.Tensor,
        vggt_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward projection and cosine alignment loss calculation.
        """
        vlm_proj = self.project_vlm(vlm_visual_tokens)
        loss = self.compute_align_loss_cosine(vlm_proj, vggt_features, mask=mask)
        return loss
