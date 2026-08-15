#!/usr/bin/env python3
"""
Geo-JEPA Diagnostics & Regression Test Suite.

Runs all 3 diagnostics:
1. Canonicalization Invariance Test (Zero-drift under camera rotation/translation)
2. Depth Probe on Latent Action Tokens (Measuring downstream geometric grounding)
3. Attention Focus & Entropy Analysis (Quantifying object concentration)
"""

import sys
import unittest
import torch

from tests.test_canonicalization import TestCanonicalization
from geo_jepa.diagnostics.depth_probe_action_tokens import ActionTokenDepthProbe, eval_probe
from geo_jepa.diagnostics.attention_visualizer import compute_attention_entropy


def run_diagnostics():
    print("=" * 70)
    print(" Geo-JEPA Automated Diagnostics & Regression Suite")
    print("=" * 70)
    
    # 1. Canonicalization Regression Test
    print("\n[1/3] Running Canonicalization Regression Test...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCanonicalization)
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    assert res.wasSuccessful(), "Canonicalization regression test failed!"

    # 2. Depth Probe Sanity Check
    print("\n[2/3] Running Latent Action Token Depth Probe Check...")
    probe = ActionTokenDepthProbe(action_dim=2048, num_action_tokens=16)
    mock_tokens = torch.randn(4, 16, 2048)
    depth_out = probe(mock_tokens)
    print(f"  Probe input:  {mock_tokens.shape}")
    print(f"  Probe output: {depth_out.shape} (Metric depth map)")
    assert depth_out.shape == (4, 1, 256, 256), "Depth probe output shape mismatch!"
    print("  Depth probe architecture sanity check: PASSED")

    # 3. Attention Visualization & Entropy Check
    print("\n[3/3] Running Attention Entropy & Focus Check...")
    H_p, W_p = 16, 16
    y, x = torch.meshgrid(torch.arange(H_p), torch.arange(W_p), indexing="ij")
    focused_attn = torch.exp(-((x - 8.0)**2 + (y - 8.0)**2) / 4.0)
    focused_attn = focused_attn / focused_attn.sum()
    diffuse_attn = torch.ones(H_p, W_p) / (H_p * W_p)
    
    e_focused = compute_attention_entropy(focused_attn).item()
    e_diffuse = compute_attention_entropy(diffuse_attn).item()
    print(f"  Focused attention entropy:  {e_focused:.4f}")
    print(f"  Diffuse attention entropy:  {e_diffuse:.4f}")
    assert e_focused < e_diffuse, "Entropy calculation error!"
    print("  Attention entropy diagnostic: PASSED")

    print("\n" + "=" * 70)
    print(" ALL GEO-JEPA DIAGNOSTICS & REGRESSIONS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostics()
