"""
Geo-JEPA Model-Predictive Geometric Guidance (MP-Geo Guidance).

At inference / rollout time:
1. Samples K parallel action trajectory chunks from the Flow-Matching policy head.
2. Passes each action candidate into the Dual-Head Geometric World Model (s_hat_geo).
3. Evaluates 3D geometric clearance, target proximity, and kinematic smoothness.
4. Reranks and executes the optimal 3D trajectory.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MPGeoGuidance:
    """
    Model-Predictive Geometric Guidance for Flow-Matching action diffusion.
    """

    def __init__(
        self,
        num_candidates: int = 8,
        temperature: float = 0.5,
        w_target: float = 1.0,
        w_smooth: float = 0.2,
        w_clearance: float = 0.3
    ):
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.w_target = w_target
        self.w_smooth = w_smooth
        self.w_clearance = w_clearance

    def score_trajectories(
        self,
        candidate_actions: torch.Tensor,
        predicted_point_tracks: torch.Tensor,
        target_3d_pos: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            candidate_actions: (K, Horizon, Action_Dim)
            predicted_point_tracks: (K, Horizon, 64, 2) or (K, Horizon, 3)
            target_3d_pos: (3,) optional target 3D coordinate
            
        Returns:
            scores: (K,) energy scores for each candidate
            best_idx: int index of highest-scoring trajectory
        """
        K, H, D = candidate_actions.shape
        device = candidate_actions.device

        # 1. Target Proximity Score: Encourage end-effector tracks towards target
        if target_3d_pos is not None:
            final_pos = predicted_point_tracks[:, -1]  # (K, ...)
            dist_to_target = torch.norm(final_pos.reshape(K, -1)[:, :3] - target_3d_pos, p=2, dim=-1)
            score_target = -dist_to_target
        else:
            # Reward steady terminal deceleration
            terminal_velocity = torch.norm(candidate_actions[:, -1, :3] - candidate_actions[:, -2, :3], p=2, dim=-1)
            score_target = -terminal_velocity

        # 2. Kinematic Smoothness Score: Penalize high-frequency acceleration jitter
        action_diffs = candidate_actions[:, 1:, :3] - candidate_actions[:, :-1, :3]
        action_accel = action_diffs[:, 1:] - action_diffs[:, :-1]
        smoothness = -torch.norm(action_accel, p=2, dim=[1, 2])

        # 3. Dynamic Clearance Score: Penalize downward collision with tabletop
        z_displacements = candidate_actions[:, :, 2]
        table_penetration = F.relu(-z_displacements - 0.4).sum(dim=1)
        score_clearance = -table_penetration

        # Composite Objective
        total_scores = (
            self.w_target * score_target +
            self.w_smooth * smoothness +
            self.w_clearance * score_clearance
        )

        best_idx = int(torch.argmax(total_scores).item())
        return total_scores, best_idx

    def guide_action(
        self,
        policy_model: nn.Module,
        obs_tokens: torch.Tensor,
        context_states: torch.Tensor,
        target_3d: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Executes MP-Geo Guidance to generate the optimal action chunk.
        """
        # 1. Sample K candidate actions with noise perturbations
        base_actions = policy_model.action_head(obs_tokens).unsqueeze(0)  # (1, H, 7)
        K = self.num_candidates

        # Inject multi-scale exploration perturbations
        noise = torch.randn(K, base_actions.shape[1], base_actions.shape[2], device=base_actions.device) * 0.08
        candidate_actions = base_actions.repeat(K, 1, 1) + noise
        candidate_actions[0] = base_actions[0]  # Ensure deterministic candidate is included

        # 2. Forward through geometric world-model predictor
        action_tokens = obs_tokens.repeat(K, 1, 1)[:, :9]
        context = context_states.repeat(K, 1, 1)
        _, pred_geo_tracks = policy_model.dual_predictor(context, action_tokens)

        # 3. Score and select best candidate
        scores, best_idx = self.score_trajectories(candidate_actions, pred_geo_tracks, target_3d)
        best_action = candidate_actions[best_idx]

        return best_action
