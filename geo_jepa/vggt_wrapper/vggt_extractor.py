"""
VGGT Feature Extractor and Sliding-Window Inference Wrapper.

Extracts:
1. Backbone visual tokens (pre-head features from Aggregator) for geometric alignment.
2. Depth maps and confidence.
3. 3D point maps canonicalized into anchor coordinates (frame 0).
4. Camera extrinsics [R | t] and intrinsics K.
5. Point tracking displacements.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from geo_jepa.vggt_wrapper.canonicalization import (
    decode_vggt_pose_encoding,
    depth_to_camera_coordinates,
    camera_to_world_coordinates,
    canonicalize_point_map,
    compute_point_track_displacements,
)


class VGGTFeatureExtractor(nn.Module):
    """
    Wrapper around VGGT for temporal sliding-window feature extraction and canonicalization.
    """

    def __init__(
        self,
        vggt_model: Optional[nn.Module] = None,
        pretrained_path: Optional[str] = None,
        layer_idx: int = -1,
        feature_only: bool = False,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.layer_idx = layer_idx
        self.feature_only = feature_only

        if vggt_model is not None:
            self.vggt = vggt_model
        elif pretrained_path is not None:
            from vggt.models.vggt import VGGT
            self.vggt = VGGT.from_pretrained(pretrained_path)
        else:
            from vggt.models.vggt import VGGT
            # Default initialization
            self.vggt = VGGT()

        self.vggt.to(self.device)
        self.vggt.eval()
        for param in self.vggt.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_temporal_window(
        self,
        video_tensor: torch.Tensor,
        window_size: Optional[int] = None,
        query_points: Optional[torch.Tensor] = None,
        canonicalize_to_first_frame: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Run VGGT over a temporal sliding window or full sequence of frames.

        Args:
            video_tensor: Input tensor of shape (B, S, 3, H, W) or (S, 3, H, W) with RGB values in [0, 1].
            window_size: If specified, truncates or slices the sequence to window_size.
            query_points: Optional query points for track head (B, N, 2) in pixel coords.
            canonicalize_to_first_frame: If True, transforms 3D point maps into frame 0's coordinate system.

        Returns:
            Dict containing:
                - "backbone_latents": (B, S, num_patches, 2048) pre-head transformer tokens (patch tokens only)
                - "depth": (B, S, H, W, 1) metric depth
                - "depth_conf": (B, S, H, W) depth confidence
                - "world_points": (B, S, H, W, 3) raw or canonicalized 3D points
                - "extrinsics": (B, S, 3, 4) camera extrinsics [R | t]
                - "intrinsics": (B, S, 3, 3) camera intrinsics K
                - "track_displacements": (B, S - 1, N, 2) if query points provided
        """
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(0)  # (1, S, 3, H, W)

        if window_size is not None and video_tensor.shape[1] > window_size:
            video_tensor = video_tensor[:, :window_size]

        video_tensor = video_tensor.to(self.device)
        B, S, C, H, W = video_tensor.shape

        # Step 1: Run Aggregator to get intermediate backbone tokens
        aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(video_tensor)

        # Extract backbone latents at target layer (default last layer -1)
        # aggregated_tokens_list is a list of tensors [B, S, P, 2*embed_dim] (e.g. 2048 dim)
        selected_layer_tokens = aggregated_tokens_list[self.layer_idx]
        if selected_layer_tokens is None:
            # Fallback to last available cached token tensor
            valid_tokens = [t for t in aggregated_tokens_list if t is not None]
            selected_layer_tokens = valid_tokens[-1]

        # Slice away camera & register tokens (first patch_start_idx tokens)
        backbone_spatial_latents = selected_layer_tokens[:, :, patch_start_idx:, :]

        results = {
            "backbone_latents": backbone_spatial_latents,
            "patch_start_idx": patch_start_idx,
        }

        # Step 2: Camera pose head
        extrinsics, intrinsics = None, None
        if hasattr(self.vggt, "camera_head") and self.vggt.camera_head is not None:
            pose_enc_list = self.vggt.camera_head(aggregated_tokens_list)
            pose_enc = pose_enc_list[-1]  # (B, S, 9)
            extrinsics, intrinsics = decode_vggt_pose_encoding(pose_enc, image_size_hw=(H, W))
            results["pose_enc"] = pose_enc
            results["extrinsics"] = extrinsics
            results["intrinsics"] = intrinsics

        # Step 3: Depth head
        depth = None
        if hasattr(self.vggt, "depth_head") and self.vggt.depth_head is not None:
            depth, depth_conf = self.vggt.depth_head(
                aggregated_tokens_list, images=video_tensor, patch_start_idx=patch_start_idx
            )
            results["depth"] = depth
            results["depth_conf"] = depth_conf

        # Step 4: 3D Point head & Canonicalization
        if hasattr(self.vggt, "point_head") and self.vggt.point_head is not None:
            pts3d, pts3d_conf = self.vggt.point_head(
                aggregated_tokens_list, images=video_tensor, patch_start_idx=patch_start_idx
            )
            if canonicalize_to_first_frame and extrinsics is not None:
                # Canonicalize world points to frame 0 coordinates
                anchor_extri = extrinsics[:, 0]  # (B, 3, 4)
                pts3d = canonicalize_point_map(pts3d, anchor_extrinsics=anchor_extri)
            results["world_points"] = pts3d
            results["world_points_conf"] = pts3d_conf
        elif depth is not None and extrinsics is not None and intrinsics is not None:
            # Reconstruct and canonicalize 3D point map from depth
            pts_cam = depth_to_camera_coordinates(depth, intrinsics)
            pts_world = camera_to_world_coordinates(pts_cam, extrinsics)
            if canonicalize_to_first_frame:
                anchor_extri = extrinsics[:, 0]
                pts3d = canonicalize_point_map(pts_world, anchor_extrinsics=anchor_extri)
            else:
                pts3d = pts_world
            results["world_points"] = pts3d

        # Step 5: Track head & Displacements
        if (
            hasattr(self.vggt, "track_head")
            and self.vggt.track_head is not None
            and query_points is not None
        ):
            if query_points.dim() == 2:
                query_points = query_points.unsqueeze(0)
            track_list, vis, conf = self.vggt.track_head(
                aggregated_tokens_list,
                images=video_tensor,
                patch_start_idx=patch_start_idx,
                query_points=query_points.to(self.device),
            )
            tracks = track_list[-1]  # (B, S, N, 2)
            results["tracks"] = tracks
            results["track_vis"] = vis
            results["track_conf"] = conf
            results["track_displacements"] = compute_point_track_displacements(tracks)

        return results
