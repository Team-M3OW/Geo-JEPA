"""
Attention Visualization Diagnostic Tool for Geo-JEPA.

Diagnostic Tool 2:
Extracts and visualizes cross-attention from <action> / <latent_i> tokens to
image patch tokens, enabling comparison of attention focus (task-relevant
objects & gripper vs background) before and after geometric forcing.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def compute_attention_entropy(attn_map: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Shannon entropy of the attention distribution.
    Lower entropy indicates sharp, concentrated attention on specific objects/regions.
    
    Args:
        attn_map: Attention tensor of shape (..., H_patches, W_patches) normalized to sum to 1.
    Returns:
        entropy: Scalar or batch of scalar entropy values.
    """
    flat_attn = attn_map.flatten(start_dim=-2)  # (..., N_patches)
    entropy = -torch.sum(flat_attn * torch.log(flat_attn + eps), dim=-1)
    return entropy


def compute_foreground_concentration(
    attn_map: torch.Tensor,
    foreground_mask: torch.Tensor
) -> float:
    """
    Compute percentage of action token attention focused inside the foreground/manipulation mask.
    
    Args:
        attn_map: 2D attention heatmap (H, W) normalized to sum to 1.
        foreground_mask: Binary mask (H, W) indicating object / gripper regions.
    """
    if attn_map.shape != foreground_mask.shape:
        attn_resized = F.interpolate(
            attn_map.unsqueeze(0).unsqueeze(0),
            size=foreground_mask.shape[-2:],
            mode="bilinear",
            align_corners=True
        ).squeeze()
    else:
        attn_resized = attn_map
        
    fg_attn = (attn_resized * foreground_mask.float()).sum().item()
    return float(fg_attn)


def render_attention_heatmap(
    image_rgb: np.ndarray,
    attn_grid: np.ndarray,
    alpha: float = 0.6,
    colormap: str = "jet"
) -> np.ndarray:
    """
    Overlay a 2D attention map onto an RGB image.
    
    Args:
        image_rgb: (H, W, 3) uint8 numpy array in range [0, 255]
        attn_grid: (H_p, W_p) float attention values
        alpha: Blending weight for attention heatmap
    Returns:
        overlay: (H, W, 3) uint8 blended image
    """
    H, W = image_rgb.shape[:2]
    
    # Normalize attention map to [0, 1]
    attn_norm = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
    
    # Resize attention to full image dimensions
    attn_pil = Image.fromarray((attn_norm * 255).astype(np.uint8)).resize((W, H), resample=Image.BICUBIC)
    attn_resized = np.array(attn_pil) / 255.0  # (H, W)
    
    # Simple colormap mapping (Blue -> Cyan -> Yellow -> Red)
    heatmap = np.zeros((H, W, 3), dtype=np.float32)
    heatmap[..., 0] = np.clip(1.5 * attn_resized - 0.5, 0.0, 1.0)        # Red
    heatmap[..., 1] = np.clip(1.0 - 2.0 * np.abs(attn_resized - 0.5), 0.0, 1.0) # Green
    heatmap[..., 2] = np.clip(1.5 - 1.5 * attn_resized, 0.0, 1.0)        # Blue
    
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    blended = (image_rgb * (1.0 - alpha) + heatmap_uint8 * alpha).astype(np.uint8)
    return blended


class ActionAttentionExtractor:
    """
    Extracts attention matrices between action tokens and visual tokens from model layers.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.attention_maps = []

    def extract_action_to_image_attention(
        self,
        attn_matrix: torch.Tensor,
        action_indices: torch.Tensor,
        image_indices: torch.Tensor,
        patch_hw: Tuple[int, int] = (16, 16)
    ) -> torch.Tensor:
        """
        Extract cross-attention slice from action tokens to image patch tokens.
        
        Args:
            attn_matrix: (B, num_heads, seq_len, seq_len)
            action_indices: 1D tensor of action token sequence positions
            image_indices: 1D tensor of image token sequence positions
            patch_hw: (H_p, W_p) spatial dimensions of image patches
            
        Returns:
            attn_spatial: (B, num_heads, len(action_indices), H_p, W_p)
        """
        B, num_heads, seq_len, _ = attn_matrix.shape
        H_p, W_p = patch_hw
        
        # Slice [action_indices, image_indices]
        action_to_img = attn_matrix[:, :, action_indices][:, :, :, image_indices]  # (B, num_heads, K, N_patches)
        
        # Softmax normalize over image patches
        action_to_img_norm = F.softmax(action_to_img, dim=-1)
        
        # Reshape to 2D spatial grid
        attn_spatial = action_to_img_norm.view(B, num_heads, len(action_indices), H_p, W_p)
        return attn_spatial


if __name__ == "__main__":
    print("[Geo-JEPA Diagnostic 2] Testing Attention Visualizer Module...")
    
    # Mock image and attention grid
    mock_image = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    
    # Concentrated Gaussian attention vs diffuse attention
    H_p, W_p = 16, 16
    y, x = np.ogrid[:H_p, :W_p]
    sharp_attn = np.exp(-((x - 8)**2 + (y - 8)**2) / 4.0)
    sharp_attn /= sharp_attn.sum()
    
    diffuse_attn = np.ones((H_p, W_p)) / (H_p * W_p)
    
    sharp_tensor = torch.tensor(sharp_attn)
    diffuse_tensor = torch.tensor(diffuse_attn)
    
    ent_sharp = compute_attention_entropy(sharp_tensor).item()
    ent_diffuse = compute_attention_entropy(diffuse_tensor).item()
    
    print(f"  Sharp Attention Entropy (Target):   {ent_sharp:.4f} (lower is sharper)")
    print(f"  Diffuse Attention Entropy (Baseline): {ent_diffuse:.4f}")
    assert ent_sharp < ent_diffuse, "Sharp attention must have lower entropy"
    
    overlay = render_attention_heatmap(mock_image, sharp_attn)
    print(f"  Rendered Overlay Shape: {overlay.shape}, Dtype: {overlay.dtype}")
    print("Attention Visualization Diagnostic verified successfully.")
