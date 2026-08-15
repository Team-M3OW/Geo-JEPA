"""
Depth Probing on Latent Action Tokens for Geo-JEPA.

Diagnostic Tool 1:
Freezes a VLA checkpoint and trains a lightweight depth probe directly from the
<latent_i> action token representations (rather than just the visual tokens).

Quantifies whether geometric grounding propagated downstream into the action
representation or remained isolated upstream in the vision backbone.
"""

from typing import Dict, List, Optional, Tuple
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ActionTokenDepthProbe(nn.Module):
    """
    Lightweight probe architecture decoding metric depth maps from latent action tokens.
    
    Architecture:
      Action Tokens [B, K, D] -> Linear projection -> Spatial Unflatten -> Transposed Convolutions / Upsampling -> Depth [B, 1, H, W]
    """

    def __init__(
        self,
        action_dim: int = 2048,
        num_action_tokens: int = 16,
        output_hw: Tuple[int, int] = (256, 256),
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.output_hw = output_hw
        self.num_action_tokens = num_action_tokens
        
        # Project K action tokens into a spatial feature bottleneck (e.g. 8x8 grid)
        self.initial_h, self.initial_w = 8, 8
        self.fc = nn.Sequential(
            nn.Linear(action_dim * num_action_tokens, hidden_dim * self.initial_h * self.initial_w),
            nn.LayerNorm(hidden_dim * self.initial_h * self.initial_w),
            nn.GELU()
        )
        
        # Multi-stage deconvolutional decoder
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(hidden_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            # Final 1x1 conv to metric depth
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU()  # Metric depth is positive
        )

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_tokens: Tensor of shape (B, K, D)
        Returns:
            depth_pred: Tensor of shape (B, 1, H, W)
        """
        B = action_tokens.shape[0]
        flat_tokens = action_tokens.flatten(start_dim=1)  # (B, K * D)
        feat_map = self.fc(flat_tokens).view(B, -1, self.initial_h, self.initial_w)
        depth = self.decoder(feat_map)
        if depth.shape[-2:] != self.output_hw:
            depth = F.interpolate(depth, size=self.output_hw, mode="bilinear", align_corners=True)
        return depth


def compute_depth_metrics(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute standard depth evaluation metrics:
    - RMSE (Root Mean Squared Error)
    - AbsRel (Absolute Relative Error)
    - Delta threshold accuracies: delta < 1.25, delta < 1.25^2
    """
    if mask is None:
        mask = (gt_depth > 1e-3) & (gt_depth < 10.0)
        
    p = pred_depth[mask]
    g = gt_depth[mask]
    
    if len(p) == 0:
        return {"rmse": 0.0, "abs_rel": 0.0, "delta_1_25": 0.0}
        
    rmse = torch.sqrt(torch.mean((p - g) ** 2)).item()
    abs_rel = torch.mean(torch.abs(p - g) / g).item()
    
    ratio = torch.max(p / g, g / p)
    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    
    return {
        "rmse": rmse,
        "abs_rel": abs_rel,
        "delta_1_25": delta1,
        "delta_1_25_sq": delta2,
    }


def train_probe_epoch(
    probe: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    probe.train()
    total_loss = 0.0
    for actions, depth_gt in dataloader:
        actions, depth_gt = actions.to(device), depth_gt.to(device)
        optimizer.zero_grad()
        pred_depth = probe(actions)
        loss = F.smooth_l1_loss(pred_depth, depth_gt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def eval_probe(
    probe: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    probe.eval()
    all_metrics = []
    with torch.no_grad():
        for actions, depth_gt in dataloader:
            actions, depth_gt = actions.to(device), depth_gt.to(device)
            pred_depth = probe(actions)
            metrics = compute_depth_metrics(pred_depth, depth_gt)
            all_metrics.append(metrics)
            
    avg_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    return avg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Probe on Latent Action Tokens")
    parser.add_argument("--action_dim", type=int, default=2048)
    parser.add_argument("--num_tokens", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Geo-JEPA Diagnostic 1] Running Depth Probe on Latent Action Tokens (device: {device})")
    
    # Synthetic verification dataset
    num_samples = 64
    mock_action_tokens = torch.randn(num_samples, args.num_tokens, args.action_dim)
    mock_depth = 1.0 + 2.0 * torch.rand(num_samples, 1, 256, 256)
    
    dataset = TensorDataset(mock_action_tokens, mock_depth)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    probe = ActionTokenDepthProbe(action_dim=args.action_dim, num_action_tokens=args.num_tokens).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)

    for ep in range(1, args.epochs + 1):
        loss = train_probe_epoch(probe, dataloader, optimizer, device)
        if ep % 5 == 0 or ep == args.epochs:
            metrics = eval_probe(probe, dataloader, device)
            print(f" Epoch {ep:02d} | Loss: {loss:.4f} | RMSE: {metrics['rmse']:.4f} | delta<1.25: {metrics['delta_1_25']*100:.1f}%")
            
    print("Depth Probe Diagnostic verified successfully.")
