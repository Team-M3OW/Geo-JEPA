# Geo-JEPA Evaluation & Ablation Scaffolding

This document defines the evaluation protocol and 4-way ablation matrix to benchmark Geo-JEPA against published VLA-JEPA and Spatial-Forcing results.

---

## 1. The 4-Way Ablation Matrix

All 4 runs are trained with the exact same data mixture and optimizer hyperparameters, varying only the geometric grounding components:

| Configuration ID | Alignment Loss $\mathcal{L}_{\text{geo}}$ (Phase 2) | Alignment Weight $\alpha$ | Prediction Head $\mathcal{L}_{\text{WM}}^{\text{geo}}$ (Phase 3) | Geometric WM Weight $\gamma$ | Config File |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **(a) Baseline VLA-JEPA** | ❌ Disabled | $\alpha = 0.0$ | ❌ Disabled | $\gamma = 0.0$ | [`configs/baseline_vla_jepa.yaml`](file:///home/kavinder/Geo-JEPA/configs/baseline_vla_jepa.yaml) |
| **(b) Geo-Align Only** | ✅ Enabled (Layer 24) | $\alpha = 0.5$ (Warmup) | ❌ Disabled | $\gamma = 0.0$ | [`configs/ablation_geo_align_only.yaml`](file:///home/kavinder/Geo-JEPA/configs/ablation_geo_align_only.yaml) |
| **(c) Geo-Pred Only** | ❌ Disabled | $\alpha = 0.0$ | ✅ Enabled (Dual-Head) | $\gamma = 0.1$ (Warmup) | [`configs/ablation_geo_pred_only.yaml`](file:///home/kavinder/Geo-JEPA/configs/ablation_geo_pred_only.yaml) |
| **(d) Full Geo-JEPA** | ✅ Enabled (Layer 24) | $\alpha = 0.5$ (Warmup) | ✅ Enabled (Dual-Head) | $\gamma = 0.1$ (Warmup) | [`configs/full_geo_jepa.yaml`](file:///home/kavinder/Geo-JEPA/configs/full_geo_jepa.yaml) |

---

## 2. Benchmark Evaluation Protocol

### 2.1 Primary Fine-Tuning & Evaluation: LIBERO Suites
Evaluate policy success rate (%) across 10 evaluation seeds (500 rollouts total):

| Benchmark Suite | Tasks | Description | Metric |
| :--- | :---: | :--- | :---: |
| **LIBERO-Spatial** | 10 | Spatial relationship and geometric arrangement tasks | Success Rate (%) |
| **LIBERO-Object** | 10 | Multi-object discrimination and manipulation | Success Rate (%) |
| **LIBERO-Goal** | 10 | Target goal state achievement | Success Rate (%) |
| **LIBERO-10 (Long)** | 10 | Multi-stage sequential manipulation | Success Rate (%) |
| **LIBERO Average** | 40 | Overall Standard Suite Average | Success Rate (%) |

### 2.2 Robustness Stress-Testing: LIBERO-Plus Perturbations
LIBERO-Plus introduces systematic out-of-distribution environmental shifts:

1. **Camera Perturbation (`Camera`)** ⭐: Camera pitch, yaw, and FOV shifts testing view invariance.
2. **Layout Perturbation (`Layout`)** ⭐: Object placement perturbations testing 3D spatial coordinate invariance.
3. **Lighting Perturbation (`Lighting`)**: Photometric illumination shifts testing depth/geometry invariance.
4. **Language Perturbation (`Language`)**: Instruction paraphrasing variations.

### 2.3 Secondary Robustness Benchmark: RoboTwin
- **RoboTwin (Hard Split)**: Bimanual domain-randomized manipulation testing robust visual token grounding under heavy camera and lighting variations.

---

## 3. Results Comparison Template

| Method / Config | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-10 | Average | LIBERO-Plus (Camera) | LIBERO-Plus (Layout) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *OpenVLA (Published)* | 84.8% | 88.6% | 79.2% | 53.6% | 76.6% | 42.1% | 51.3% |
| *Spatial-Forcing (Published)*| 91.2% | 93.4% | 86.8% | 68.2% | 84.9% | 67.4% | 72.8% |
| *VLA-JEPA (Published)* | 92.6% | 94.0% | 88.4% | 74.2% | 87.3% | 61.2% | 68.5% |
| **(a) Baseline VLA-JEPA (Ours)** | — | — | — | — | — | — | — |
| **(b) Geo-Align Only (Ours)** | — | — | — | — | — | — | — |
| **(c) Geo-Pred Only (Ours)** | — | — | — | — | — | — | — |
| **(d) Full Geo-JEPA (Ours)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
