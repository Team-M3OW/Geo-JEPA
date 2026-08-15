"""
Unit Tests for Phase 2: Geometric Alignment Projector, Hook, and Warmup Schedulers.
"""

import unittest
import torch
import torch.nn as nn

from geo_jepa.models.align_projector import (
    AlignProjector,
    apply_pos_embed,
    interpolate_pooling,
    create_uv_grid,
    position_grid_to_embed,
)
from geo_jepa.models.qwen_alignment_hook import QwenGeometricAlignmentHook
from geo_jepa.training.warmup_scheduler import (
    CoefficientWarmupScheduler,
    GeoJEPALossScheduler,
)


class TestAlignProjector(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_align_projector_forward_and_gradients(self):
        """
        Verify projection dimensions, cosine loss computation, and gradient backprop.
        """
        B, N_tokens, vlm_dim, vggt_dim = 2, 256, 2048, 1024
        
        projector = AlignProjector(
            vlm_dim=vlm_dim,
            vggt_dim=vggt_dim,
            align_loss_type="cosine"
        ).to(self.device)

        # Mock VLM visual tokens (requires grad)
        vlm_tokens = torch.randn(B, N_tokens, vlm_dim, device=self.device, requires_grad=True)
        # Mock frozen VGGT target (2 * 1024 = 2048)
        vggt_target = torch.randn(B, N_tokens, 2 * vggt_dim, device=self.device)

        loss = projector(vlm_tokens, vggt_target)

        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertGreater(loss.item(), 0.0)

        # Verify gradient flow to VLM visual tokens
        loss.backward()
        self.assertIsNotNone(vlm_tokens.grad)
        self.assertGreater(vlm_tokens.grad.abs().sum().item(), 0.0)
        print(f"[AlignProjector Test] Loss: {loss.item():.4f}, VLM Token Gradient Norm: {vlm_tokens.grad.norm().item():.4f}")

    def test_uv_positional_embeddings(self):
        """
        Verify that UV positional embedding tensor matches expected dimensions and adds spatial signal.
        """
        B_N, D, H_p, W_p = 2, 2048, 37, 37
        x = torch.zeros(B_N, D, H_p, W_p, device=self.device)
        x_with_pe = apply_pos_embed(x, img_w=518, img_h=518, ratio=0.1)

        self.assertEqual(x_with_pe.shape, (B_N, D, H_p, W_p))
        # Verify corner tokens have different embeddings (spatial asymmetry)
        corner_top_left = x_with_pe[0, :, 0, 0]
        corner_bottom_right = x_with_pe[0, :, -1, -1]
        diff = torch.norm(corner_top_left - corner_bottom_right).item()
        self.assertGreater(diff, 0.01, "Positional embeddings must distinguish spatial positions")
        print(f"[Positional Embed Test] Top-Left vs Bottom-Right Corner Difference: {diff:.4f}")

    def test_qwen_alignment_hook(self):
        """
        Verify visual token extraction and end-to-end geometric loss computation through Qwen hook.
        """
        B, seq_len, num_layers, D_vlm = 2, 600, 32, 2048
        target_layer = 24  # 75% depth
        image_token_id = 151655
        
        # Build mock Qwen hidden states
        hidden_states = tuple(
            torch.randn(B, seq_len, D_vlm, device=self.device, requires_grad=(l == target_layer))
            for l in range(num_layers)
        )
        
        # Construct input_ids with 256 visual tokens in the middle
        input_ids = torch.randint(100, 1000, (B, seq_len), device=self.device)
        input_ids[:, 50:306] = image_token_id  # 256 image tokens

        # Mock current-timestep VGGT features: (B, 1, 37*37, 2048)
        vggt_features = torch.randn(B, 1, 37 * 37, 2048, device=self.device)

        hook = QwenGeometricAlignmentHook(
            vlm_dim=D_vlm,
            vggt_dim=1024,
            alignment_layer_idx=target_layer,
            image_token_id=image_token_id
        ).to(self.device)

        geo_loss = hook.compute_geometric_loss(
            hidden_states=hidden_states,
            input_ids=input_ids,
            vggt_current_features=vggt_features,
            vggt_patch_hw=(37, 37),
            vggt_img_hw=(518, 518)
        )

        self.assertTrue(torch.is_tensor(geo_loss))
        self.assertEqual(geo_loss.dim(), 0)
        geo_loss.backward()
        self.assertIsNotNone(hidden_states[target_layer].grad)
        print(f"[Qwen Alignment Hook Test] L_geo: {geo_loss.item():.4f}, Layer {target_layer} Gradient Norm: {hidden_states[target_layer].grad.norm().item():.4f}")

    def test_coefficient_warmup_scheduler(self):
        """
        Verify that warmup ramps from start_val (0.0) to target_val (0.5) smoothly.
        """
        scheduler = CoefficientWarmupScheduler(
            target_val=0.5,
            warmup_steps=2000,
            schedule_type="linear"
        )
        
        self.assertEqual(scheduler.get_val(0), 0.0)
        self.assertAlmostEqual(scheduler.get_val(1000), 0.25, places=5)
        self.assertEqual(scheduler.get_val(2000), 0.5)
        self.assertEqual(scheduler.get_val(5000), 0.5)

        geo_manager = GeoJEPALossScheduler(
            alpha_target=0.5,
            alpha_warmup_steps=2000,
            beta=0.1,
            gamma_target=0.1,
            gamma_warmup_steps=2000
        )
        coeffs_step_0 = geo_manager.get_coefficients(0)
        coeffs_step_1000 = geo_manager.get_coefficients(1000)
        coeffs_step_2000 = geo_manager.get_coefficients(2000)

        self.assertEqual(coeffs_step_0["alpha"], 0.0)
        self.assertAlmostEqual(coeffs_step_1000["alpha"], 0.25, places=5)
        self.assertEqual(coeffs_step_2000["alpha"], 0.5)
        self.assertEqual(coeffs_step_2000["beta"], 0.1)
        self.assertEqual(coeffs_step_2000["gamma"], 0.1)
        print(f"[Warmup Scheduler Test] Step 0: {coeffs_step_0}, Step 1000: {coeffs_step_1000}, Step 2000: {coeffs_step_2000}")


if __name__ == "__main__":
    unittest.main()
