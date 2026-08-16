"""
Unit Tests for MultiRayGraspBundleProjector and MP-Geo Guidance.
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/home/kavinder/Geo-JEPA")

from geo_jepa.models.action_ray_head import MultiRayGraspBundleProjector
from geo_jepa.models.mp_geo_guidance import MPGeoGuidance


def test_multi_ray_grasp_bundle():
    B, N_tokens, D = 4, 9, 1024
    action_tokens = torch.randn(B, N_tokens, D, requires_grad=True)

    projector = MultiRayGraspBundleProjector(action_dim=D, hidden_dim=256, num_cone_rays=4)
    bundle = projector(action_tokens)

    # Assert shapes
    assert bundle["ray_left"].shape == (B, 3)
    assert bundle["ray_palm"].shape == (B, 3)
    assert bundle["ray_right"].shape == (B, 3)
    assert bundle["ray_cone"].shape == (B, 4, 3)
    assert bundle["distances"].shape == (B, 3)
    assert bundle["aperture"].shape == (B, 1)

    # Assert unit vector normalization
    for k in ["ray_left", "ray_palm", "ray_right"]:
        norms = torch.norm(bundle[k], p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Compute loss against ground truth contact points
    gt_left = torch.tensor([[0.18, 0.40, 0.10], [0.08, -0.30, 0.20], [-0.22, 0.10, 0.0], [0.28, 0.0, 0.40]])
    gt_right = torch.tensor([[0.22, 0.40, 0.10], [0.12, -0.30, 0.20], [-0.18, 0.10, 0.0], [0.32, 0.0, 0.40]])
    gt_center = torch.tensor([[0.20, 0.40, 0.10], [0.10, -0.30, 0.20], [-0.20, 0.10, 0.0], [0.30, 0.0, 0.40]])
    gt_gripper = torch.tensor([[0.0, 0.0, 0.50], [0.0, 0.0, 0.50], [0.0, 0.0, 0.50], [0.0, 0.0, 0.50]])
    gt_aperture = torch.tensor([[0.04], [0.05], [0.04], [0.06]])

    loss_dict = projector.compute_bundle_loss(
        bundle, gt_left, gt_right, gt_center, gt_gripper, gt_aperture
    )
    loss = loss_dict["loss_bundle_total"]

    loss.backward()
    assert action_tokens.grad is not None
    assert not torch.isnan(loss)
    print(f"MultiRayGraspBundleProjector Unit Test PASSED! Bundle Loss: {loss.item():.4f}, Mean Cos Sim: {loss_dict['mean_cos_sim'].item():.4f}")


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
    test_multi_ray_grasp_bundle()
    test_mp_geo_guidance()
