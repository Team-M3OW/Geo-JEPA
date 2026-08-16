"""
Unit Tests for ActionRayProjector and MP-Geo Guidance.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/home/kavinder/Geo-JEPA")

from geo_jepa.models.action_ray_head import ActionRayProjector
from geo_jepa.models.mp_geo_guidance import MPGeoGuidance


def test_action_ray_projector():
    B, N_tokens, D = 4, 9, 1024
    action_tokens = torch.randn(B, N_tokens, D, requires_grad=True)

    projector = ActionRayProjector(action_dim=D, hidden_dim=256)
    ray_dir, ray_dist = projector(action_tokens)

    # Assert shapes
    assert ray_dir.shape == (B, 3)
    assert ray_dist.shape == (B, 1)

    # Assert unit vector norm
    norms = torch.norm(ray_dir, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Compute loss against target coordinates
    target_pos_3d = torch.tensor([[0.2, 0.4, 0.1], [0.1, -0.3, 0.2], [-0.2, 0.1, 0.0], [0.3, 0.0, 0.4]])
    gripper_pos_3d = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])

    loss_dict = projector.compute_ray_loss(ray_dir, ray_dist, target_pos_3d, gripper_pos_3d)
    loss = loss_dict["loss_ray_total"]

    loss.backward()
    assert action_tokens.grad is not None
    assert not torch.isnan(loss)
    print(f"ActionRayProjector Unit Test PASSED! Total Ray Loss: {loss.item():.4f}")


def test_mp_geo_guidance():
    K, H, D = 8, 8, 7
    candidate_actions = torch.randn(K, H, D)
    pred_tracks = torch.randn(K, H, 64, 2)
    target_3d = torch.tensor([0.2, 0.3, 0.1])

    guidance = MPGeoGuidance(num_candidates=K)
    scores, best_idx = guidance.score_trajectories(candidate_actions, pred_tracks, target_3d)

    assert scores.shape == (K,)
    assert 0 <= best_idx < K
    assert not torch.isnan(scores).any()
    print(f"MPGeoGuidance Unit Test PASSED! Best Candidate Index: {best_idx} (Score: {scores[best_idx].item():.4f})")


if __name__ == "__main__":
    test_action_ray_projector()
    test_mp_geo_guidance()
