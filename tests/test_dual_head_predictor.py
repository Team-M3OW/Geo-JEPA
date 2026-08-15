"""
Unit Tests for Phase 3: Dual-Head World Model Predictor and Loss Composition.
"""

import unittest
import torch
import torch.nn as nn

from geo_jepa.models.dual_head_predictor import DualHeadVisionTransformerPredictor


class TestDualHeadPredictor(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_dual_head_forward_and_losses(self):
        """
        Verify that the dual-head predictor produces correct shapes for semantic and geometric outputs,
        and computes losses properly with gradient flow.
        """
        B = 2
        num_frames = 4  # T
        patch_size = 16
        img_size = (64, 64)  # Small grid for test: 4x4 = 16 patches per frame
        grid_h, grid_w = img_size[0] // patch_size, img_size[1] // patch_size  # 4, 4
        num_patches_per_frame = grid_h * grid_w  # 16
        
        embed_dim_sem = 512
        geo_target_dim = 128
        predictor_embed_dim = 256
        action_embed_dim = 512
        num_add_tokens = 3

        predictor = DualHeadVisionTransformerPredictor(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=1,
            embed_dim_semantic=embed_dim_sem,
            geo_target_dim=geo_target_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=2,
            num_heads=4,
            action_embed_dim=action_embed_dim,
            num_add_tokens=num_add_tokens
        ).to(self.device)

        # Context states for T-1 frames: [B, (T-1)*num_patches, embed_dim_sem]
        T_minus_1 = num_frames - 1  # 3
        context_len = T_minus_1 * num_patches_per_frame  # 3 * 16 = 48
        context_states = torch.randn(B, context_len, embed_dim_sem, device=self.device, requires_grad=True)

        # Action tokens: [B, T_minus_1 * num_add_tokens, action_embed_dim]
        action_tokens = torch.randn(B, T_minus_1 * num_add_tokens, action_embed_dim, device=self.device, requires_grad=True)

        # Forward pass
        pred_sem, pred_geo = predictor(context_states, action_tokens)

        self.assertEqual(pred_sem.shape, (B, context_len, embed_dim_sem))
        self.assertEqual(pred_geo.shape, (B, context_len, geo_target_dim))

        # Ground truth targets (stop-gradient)
        gt_sem = torch.randn(B, context_len, embed_dim_sem, device=self.device)
        gt_geo = torch.randn(B, context_len, geo_target_dim, device=self.device)

        loss_dict = predictor.compute_dual_wm_loss(
            pred_sem_states=pred_sem,
            gt_sem_states=gt_sem,
            pred_geo_states=pred_geo,
            gt_geo_states=gt_geo,
            gamma=0.2
        )

        self.assertIn("wm_loss_sem", loss_dict)
        self.assertIn("wm_loss_geo", loss_dict)
        self.assertIn("wm_loss_total", loss_dict)

        total_loss = loss_dict["wm_loss_total"]
        self.assertGreater(total_loss.item(), 0.0)

        # Verify gradient flow to context states and action tokens
        total_loss.backward()
        self.assertIsNotNone(context_states.grad)
        self.assertIsNotNone(action_tokens.grad)
        self.assertGreater(context_states.grad.norm().item(), 0.0)
        self.assertGreater(action_tokens.grad.norm().item(), 0.0)

        print(f"\n[Dual-Head Predictor Test]")
        print(f"  Semantic Loss:    {loss_dict['wm_loss_sem'].item():.4f}")
        print(f"  Geometric Loss:   {loss_dict['wm_loss_geo'].item():.4f}")
        print(f"  Total WM Loss:    {loss_dict['wm_loss_total'].item():.4f}")
        print(f"  Context Grad Norm: {context_states.grad.norm().item():.4f}")
        print(f"  Action Grad Norm:  {action_tokens.grad.norm().item():.4f}")


if __name__ == "__main__":
    unittest.main()
