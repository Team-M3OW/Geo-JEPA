# Geo-JEPA: Geometric-Temporal Joint Embedding Predictive Architecture

[![GitHub Repo](https://img.shields.io/badge/GitHub-Team--M3OW%2FGeo--JEPA-blue.svg)](https://github.com/Team-M3OW/Geo-JEPA)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Geo-JEPA** unites geometric spatial grounding via frozen VGGT visual geometry foundation models with leakage-free temporal dynamics via VLA-JEPA, creating an embodied representation for robotic manipulation that is robust to camera viewpoint shifts, lighting changes, and layout perturbations.

---

## 🌟 Key Architecture & Loss Formulation

Geo-JEPA optimizes a four-component objective:
$$\mathcal{L} = \mathcal{L}_{\text{FM}} + \beta \cdot \mathcal{L}_{\text{WM}}^{\text{sem}} + \gamma(t) \cdot \mathcal{L}_{\text{WM}}^{\text{geo}} + \alpha(t) \cdot \mathcal{L}_{\text{geo}}$$

1. **$\mathcal{L}_{\text{FM}}$ (Action Flow-Matching Head)**: Supervised continuous action generation conditioned on embodied latent tokens.
2. **$\mathcal{L}_{\text{geo}}$ (Mid-Depth Geometric Alignment)**: Spatial-Forcing alignment between mid-depth Qwen-VL visual tokens and frozen VGGT backbone representations at ~75% depth (Layer 24), with UV positional embeddings.
3. **$\mathcal{L}_{\text{WM}}^{\text{sem}}$ (Semantic World Model)**: Leakage-free future latent state prediction supervised by frozen V-JEPA2 targets.
4. **$\mathcal{L}_{\text{WM}}^{\text{geo}}$ (Geometric World Model)**: Dual-head prediction of canonicalized 3D point-track displacements ($\Delta \mathbf{p}_t$) and metric geometry.
5. **Progressive Warmup Schedulers**: Linear warmup schedules for $\alpha(t)$ and $\gamma(t)$ over the initial $N$ steps to ensure stable visual token formation.

---

## 📁 Repository Structure

```text
Geo-JEPA/
├── configs/
│   ├── baseline_vla_jepa.yaml        # (a) Baseline VLA-JEPA (alpha=0, gamma=0)
│   ├── ablation_geo_align_only.yaml  # (b) Phase 2 Only: Geometric Alignment (alpha=0.5)
│   ├── ablation_geo_pred_only.yaml   # (c) Phase 3 Only: Dual-Head Predictor (gamma=0.1)
│   └── full_geo_jepa.yaml            # (d) Full Geo-JEPA (alpha=0.5 + gamma=0.1 + Warmup)
├── geo_jepa/
│   ├── models/
│   │   ├── align_projector.py        # Spatial-Forcing AlignProjector + UV pos-embed pooling
│   │   ├── qwen_alignment_hook.py    # Mid-layer Qwen-VL visual token extraction & loss
│   │   ├── dual_head_predictor.py    # Dual-head ACBlock world model predictor
│   │   └── geo_jepa_framework.py     # Master Geo-JEPA framework class
│   ├── vggt_wrapper/
│   │   ├── canonicalization.py       # 3D Point map canonicalization & pose decoding
│   │   └── vggt_extractor.py         # Sliding-window [t-k, ..., t+T] feature extraction
│   ├── training/
│   │   ├── warmup_scheduler.py       # Dynamic alpha(t) & gamma(t) warmup schedulers
│   │   └── train_geo_jepa.py         # Co-training orchestrator
│   └── diagnostics/
│       ├── depth_probe_action_tokens.py # Depth probing on <latent_i> action tokens
│       ├── attention_visualizer.py      # Cross-attention heatmaps & entropy
│       └── run_all_diagnostics.py       # Automated regression test runner
├── scripts/
│   ├── cache_vggt_features.py        # Pretraining feature caching CLI
│   └── estimate_caching_footprint.py # Dataset storage footprint estimator
├── tests/
│   ├── test_canonicalization.py      # Camera motion invariance regression test
│   ├── test_align_projector.py       # Alignment projector & warmup test
│   └── test_dual_head_predictor.py   # Dual-head predictor loss & gradient test
└── docs/
    └── eval_scaffolding.md           # LIBERO & LIBERO-Plus evaluation protocol
```

---

## 🚀 Quick Start

### 1. Verification & Regression Tests
Run the complete automated regression and diagnostics suite:
```bash
python geo_jepa/diagnostics/run_all_diagnostics.py
```

### 2. Feature Caching
Cache temporal sliding-window features ($[t-k, \dots, t+T]$):
```bash
python scripts/cache_vggt_features.py \
    --dataset_name droid \
    --window_past_k 2 \
    --window_future_t 8 \
    --output_dir /media/kavinder/hdd2/geo_jepa_cache \
    --sample_limit 50
```

### 3. Launching the 4-Way Ablations
```bash
# (a) Baseline VLA-JEPA
python geo_jepa/training/train_geo_jepa.py --config configs/baseline_vla_jepa.yaml

# (b) Geometric Alignment Only (Phase 2)
python geo_jepa/training/train_geo_jepa.py --config configs/ablation_geo_align_only.yaml

# (c) Geometric Dynamics Prediction Only (Phase 3)
python geo_jepa/training/train_geo_jepa.py --config configs/ablation_geo_pred_only.yaml

# (d) Full Geo-JEPA
python geo_jepa/training/train_geo_jepa.py --config configs/full_geo_jepa.yaml
```

---

## 📊 Benchmark Scaffolding

For detailed evaluation protocols on **LIBERO-Spatial**, **LIBERO-Object**, **LIBERO-Goal**, **LIBERO-10 (Long)**, and **LIBERO-Plus** (Camera & Layout perturbations), see [`docs/eval_scaffolding.md`](docs/eval_scaffolding.md).
