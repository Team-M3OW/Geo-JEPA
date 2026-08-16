"""
Unit Test for Coupled Geometric-Action Joint Flow Head.
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/home/kavinder/Geo-JEPA")

from geo_jepa.models.coupled_geo_action_flow import CoupledGeoActionFlow


def test_coupled_geo_action_flow():
    B = 4
    cond_dim = 1024
    action_dim = 7
    geo_dim = 128
    horizon = 8

    flow_head = CoupledGeoActionFlow(
        cond_dim=cond_dim,
        action_dim=action_dim,
        geo_dim=geo_dim,
        horizon=horizon,
        hidden_dim=256,
        num_layers=3
    )

    cond = torch.randn(B, cond_dim, requires_grad=True)
    actions_gt = torch.randn(B, horizon, action_dim)
    geo_tracks_gt = torch.randn(B, horizon, geo_dim)

    # 1. Test Single Joint Flow-Matching Loss Computation
    loss_dict = flow_head.compute_flow_loss(actions_gt, geo_tracks_gt, cond)
    total_loss = loss_dict["loss_coupled_flow"]
    loss_act = loss_dict["loss_action_component"]
    loss_geo = loss_dict["loss_geo_component"]

    assert total_loss.item() > 0
    assert not torch.isnan(total_loss)

    # 2. Test Gradient Backprop
    total_loss.backward()
    assert cond.grad is not None
    print(f"Coupled Flow Loss: {total_loss.item():.4f} (Action Component: {loss_act.item():.4f}, Geo Component: {loss_geo.item():.4f})")

    # 3. Test ODE Sampling (Euler Integration from t=0 to 1)
    pred_actions, pred_geo = flow_head.sample_trajectory(cond, num_steps=4)

    assert pred_actions.shape == (B, horizon, action_dim)
    assert pred_geo.shape == (B, horizon, geo_dim)
    assert not torch.isnan(pred_actions).any()
    assert not torch.isnan(pred_geo).any()

    print(f"ODE Sampling Verified! Generated Joint Trajectories: Actions {pred_actions.shape}, Geometry {pred_geo.shape}")
    print("CoupledGeoActionFlow Unit Test PASSED successfully!")


if __name__ == "__main__":
    test_coupled_geo_action_flow()
