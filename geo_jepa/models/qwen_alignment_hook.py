"""
Qwen-VL Visual Token Alignment Hook for Geo-JEPA.

Handles:
1. Extracting intermediate hidden states at a configurable alignment layer index (e.g. 75% depth).
2. Slicing out visual patch tokens from the combined text+vision sequence using image token masks.
3. Aligning with multi-view / single-view VGGT feature maps via AlignProjector.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from geo_jepa.models.align_projector import AlignProjector, interpolate_pooling


class QwenGeometricAlignmentHook(nn.Module):
    """
    Manages intermediate layer visual token extraction from Qwen-VL backbones
    and computes the geometric alignment loss with frozen VGGT targets.
    """

    def __init__(
        self,
        vlm_dim: int = 2048,
        vggt_dim: int = 1024,
        alignment_layer_idx: int = 24,  # ~75% depth for 32-layer backbone
        image_token_id: int = 151655,   # Qwen-VL IMAGE_TOKEN_INDEX / <|image_pad|>
        use_vggt_pe: bool = True,
        pe_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.alignment_layer_idx = alignment_layer_idx
        self.image_token_id = image_token_id
        self.use_vggt_pe = use_vggt_pe
        self.pe_ratio = pe_ratio
        
        self.align_projector = AlignProjector(
            vlm_dim=vlm_dim,
            vggt_dim=vggt_dim,
            align_loss_type="cosine"
        )

    def extract_visual_tokens(
        self,
        hidden_states: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
        input_ids: torch.Tensor,
        target_layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract visual token embeddings at the specified layer index.
        
        Args:
            hidden_states: Tuple of hidden state tensors from VLM forward pass [layer_0, ..., layer_L]
            input_ids: Input token ID tensor of shape (B, seq_len)
            target_layer_idx: Layer to extract from (defaults to self.alignment_layer_idx)
            
        Returns:
            visual_tokens: Tensor of shape (B, max_vis_tokens, D_vlm)
            visual_mask: Boolean mask of shape (B, max_vis_tokens)
        """
        layer_idx = target_layer_idx if target_layer_idx is not None else self.alignment_layer_idx
        num_layers = len(hidden_states)
        
        # Safe layer indexing (supports negative indices)
        if layer_idx < 0:
            layer_idx = num_layers + layer_idx
        layer_idx = max(0, min(layer_idx, num_layers - 1))
        
        selected_hidden = hidden_states[layer_idx]  # (B, seq_len, D_vlm)
        B, seq_len, D = selected_hidden.shape
        
        # Boolean mask of image tokens
        vision_mask = (input_ids == self.image_token_id)  # (B, seq_len)
        
        if not vision_mask.any():
            # Fallback: return dummy visual tokens if no image tokens present in batch
            dummy_tokens = selected_hidden.new_zeros(B, 1, D)
            dummy_mask = torch.zeros(B, 1, dtype=torch.bool, device=input_ids.device)
            return dummy_tokens, dummy_mask
            
        # Determine number of vision tokens per sample
        counts = vision_mask.sum(dim=1)  # (B,)
        max_vis_tokens = counts.max().item()
        
        # Gather visual tokens into packed tensor (B, max_vis_tokens, D)
        visual_tokens = selected_hidden.new_zeros(B, max_vis_tokens, D)
        packed_mask = torch.zeros(B, max_vis_tokens, dtype=torch.bool, device=input_ids.device)
        
        for b in range(B):
            sample_vis_indices = vision_mask[b].nonzero(as_tuple=True)[0]
            n_tokens = len(sample_vis_indices)
            if n_tokens > 0:
                visual_tokens[b, :n_tokens] = selected_hidden[b, sample_vis_indices]
                packed_mask[b, :n_tokens] = True
                
        return visual_tokens, packed_mask

    def compute_geometric_loss(
        self,
        hidden_states: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
        input_ids: torch.Tensor,
        vggt_current_features: torch.Tensor,
        vggt_patch_hw: Tuple[int, int] = (37, 37),
        vggt_img_hw: Tuple[int, int] = (518, 518)
    ) -> torch.Tensor:
        """
        Extract VLM visual tokens and compute alignment loss against current-timestep VGGT features.
        
        Args:
            hidden_states: VLM all hidden states
            input_ids: VLM token ids
            vggt_current_features: (B, N_views, num_patches, 2048) or (B, num_patches, 2048)
            vggt_patch_hw: Spatial grid dimensions for VGGT
            vggt_img_hw: Image resolution for VGGT
            
        Returns:
            Scalar alignment loss L_geo
        """
        # Step 1: Extract VLM visual tokens at alignment layer
        vlm_vis_tokens, vis_mask = self.extract_visual_tokens(hidden_states, input_ids)
        
        if not vis_mask.any():
            return torch.tensor(0.0, device=vlm_vis_tokens.device, requires_grad=True)
            
        # Ensure VGGT features match VLM batch size and (B, N_views, S_patches, D)
        B_vlm = vlm_vis_tokens.shape[0]
        if vggt_current_features.dim() == 3:
            if vggt_current_features.shape[0] != B_vlm:
                # Features are (N_views, S_patches, D)
                vggt_current_features = vggt_current_features.unsqueeze(0).expand(B_vlm, -1, -1, -1)
            else:
                # Features are (B, S_patches, D)
                vggt_current_features = vggt_current_features.unsqueeze(1)
        elif vggt_current_features.dim() == 4:
            if vggt_current_features.shape[0] != B_vlm:
                reps = (B_vlm // vggt_current_features.shape[0]) + 1
                vggt_current_features = vggt_current_features.repeat(reps, 1, 1, 1)[:B_vlm]
            
        B, N_views, S_patches, D_vggt = vggt_current_features.shape
        target_num_tokens = vlm_vis_tokens.shape[1]
        
        # Step 2: Pool & interpolate VGGT features to match VLM visual token grid with UV positional embeddings
        vggt_pooled = interpolate_pooling(
            hidden=vggt_current_features,
            patch_hw=vggt_patch_hw,
            img_hw=vggt_img_hw,
            target_num_tokens=target_num_tokens,
            mode="bilinear",
            use_vggt_pe=self.use_vggt_pe,
            pe_ratio=self.pe_ratio
        )  # (B, target_num_tokens, D_vggt)
        
        # Step 3: Project VLM tokens and compute cosine similarity loss
        align_loss = self.align_projector(
            vlm_visual_tokens=vlm_vis_tokens,
            vggt_features=vggt_pooled,
            mask=vis_mask
        )
        
        return align_loss
