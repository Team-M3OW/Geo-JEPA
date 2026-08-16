# Geo-JEPA: Detailed System Architecture & Mathematical Foundations

This document provides an exhaustive, component-by-component architectural reference for **Geo-JEPA** (Grounding Joint-Embedding Predictive Architectures with 3D Geometric Foundation Models).

---

## 1. High-Level System Overview

Geo-JEPA integrates four foundational deep learning paradigms:
1. **Vision-Language Understanding**: Qwen2.5-VL / Qwen3-VL Vision Transformer backbone.
2. **Spatial-Forcing 3D Anchoring**: Mid-depth visual token alignment with frozen **VGGT** foundation representations.
3. **Dual-Head World-Model Forecasting**: Joint future state prediction of semantic video latents (**V-JEPA2**) and canonicalized 3D point-track displacements.
4. **Flow-Matching Action Generation**: Diffusion Transformer (DiT-B) policy head for continuous 7-DoF trajectory generation.

```
══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    GEO-JEPA COMPLETE SYSTEM ARCHITECTURE
══════════════════════════════════════════════════════════════════════════════════════════════════════

 [Dual RGB Cameras]        [Language Goal]                 [Frozen Teachers (Training Only)]
 (AgentView + WristView)   ("Pick up the bowl...")          VGGT-3D (Layer 24)    V-JEPA2 (ViT-L)
         │                         │                               │                     │
         ▼                         ▼                               │                     │
  ┌───────────────────────────────────────────┐                    │                     │
  │     Qwen-VL Vision-Language Backbone      │                    │                     │
  │  (Multi-Modal Self-Attention Transformer) │                    │                     │
  └─────┬──────────────────┬────────────┬─────┘                    │                     │
        │                  │            │                          │                     │
  Layer 24 Visual Tokens   │    <|embodied_action|>                │                     │
        │                  │            │                          │                     │
        ▼                  │            ▼                          │                     │
 ┌──────────────┐          │    ┌───────────────────┐              │                     │
 │AlignProjector│          │    │ Flow-Matching DiT │              │                     │
 │   + UV PE    │          │    │ Action Policy     │              │                     │
 └──────┬───────┘          │    └─────────┬─────────┘              │                     │
        │                  │              │                        │                     │
        ▼ (L_geo)          │              ▼ (L_FM)                 │                     │
 ┌──────────────┐          │     Predicted 7-DoF Action            │                     │
 │ Cosine Loss  │◄─────────┼───────────────────────────────────────┘                     │
 └──────────────┘          │     [Δx, Δy, Δz, Δr, Δp, Δy, Grip]                          │
                           │                                                             │
                  <|action_0...K|> (Latents)                                             │
                           │                                                             │
                           ▼                                                             │
                 ┌───────────────────┐                                                   │
                 │ Dual-Head Predictor│                                                  │
                 │  World Model (AC) │                                                   │
                 └────┬─────────┬────┘                                                   │
                      │         │                                                        │
                      ▼         ▼                                                        │
                 Head 1 (Sem)  Head 2 (Geo)                                              │
                      │         │                                                        │
                      ▼         ▼                                                        │
                 (L_WM_sem)   (L_WM_geo)                                                 │
                      │         │                                                        │
                      ▼         ▼                                                        │
                 [V-JEPA2]   [VGGT 3D Point Tracks]◄─────────────────────────────────────┘
```

---

## 2. Token Ingestion & Sequence Formatting

Input observations are tokenized into a unified multimodal sequence fed to the VLM:

$$\mathbf{T}_{\text{input}} = \big[ \mathbf{t}_{\text{sys}}, \, \mathbf{t}_{\text{img}}^{\text{agent}}, \, \mathbf{t}_{\text{img}}^{\text{wrist}}, \, \mathbf{t}_{\text{lang}}, \, \mathbf{t}_{\text{action}}^{(0 \dots K)}, \, \mathbf{t}_{\text{embodied}} \big]$$

### Special Token Hierarchy:
1. **Visual Patch Tokens ($\mathbf{t}_{\text{img}}$)**: Grid of visual patch representations tagged with `<|image_pad|>` (ID: `151655`).
2. **World-Model Conditioning Tokens ($\mathbf{t}_{\text{action}}^{(k)}$)**: Special prompt tokens `<|action_0|>` to `<|action_K|>` whose hidden states condition future world-model prediction.
3. **Embodied Policy Token ($\mathbf{t}_{\text{embodied}}$)**: Special token `<|embodied_action|>` whose final hidden state conditions the Flow-Matching action head.

---

## 3. Module 1: Spatial-Forcing Geometric Alignment (`AlignProjector`)

To prevent the visual tokens from remaining ungrounded 2D pixel statistics, Geo-JEPA extracts visual patch tokens at **Layer 24 (~75% depth)** and aligns them with **VGGT Aggregator representations**.

### 3.1 Sinusoidal 2D UV Positional Embeddings
To preserve continuous spatial layout during bilinear token resampling, we generate sinusoidal UV frequency bands:

$$\omega_k = \frac{1}{\omega_0^{2k/D}}, \quad k \in \left[0, \frac{D}{4}-1\right], \quad \omega_0 = 10000.0$$

For normalized spatial grid coordinates $(u, v) \in [0, 1] \times [0, 1]$:

$$\text{PE}(u, v) = \Big[ \sin(\omega_k u), \, \cos(\omega_k u), \, \sin(\omega_k v), \, \cos(\omega_k v) \Big] \in \mathbb{R}^D$$

The visual token representations are augmented via:

$$\tilde{\mathbf{z}}_{\text{VGGT}} = \mathbf{z}_{\text{VGGT}} + \lambda_{\text{pe}} \cdot \text{PE}(u, v), \quad \lambda_{\text{pe}} = 0.10$$

### 3.2 Dynamic Bilinear Interpolation Pooling
When reshaping spatial patches from sequence length $S = H_p \times W_p$ (e.g., $37 \times 37 = 1369$ for $518 \times 518$ images) to match the VLM token count $S_{\text{VLM}}$:

$$\mathbf{z}_{\text{pooled}} = \text{GridSample}\Big( \tilde{\mathbf{z}}_{\text{VGGT}}, \, S_{\text{target}} = S_{\text{VLM}}, \, \text{mode} = \text{"bilinear"} \Big)$$

### 3.3 Geometric Cosine Alignment Loss ($\mathcal{L}_{\text{geo}}$)
The student VLM tokens $\mathbf{z}_{\text{VLM}}^{(L24)}$ are projected through a 2-layer MLP $\phi_{\text{align}}: \mathbb{R}^{D_{\text{VLM}}} \to \mathbb{R}^{D_{\text{VGGT}}}$:

$$\mathcal{L}_{\text{geo}} = 1 - \frac{1}{N_{\text{vis}}} \sum_{i=1}^{N_{\text{vis}}} \frac{\phi_{\text{align}}(\mathbf{z}_{\text{VLM}, i}^{(L24)}) \cdot \mathbf{z}_{\text{pooled}, i}}{\big\| \phi_{\text{align}}(\mathbf{z}_{\text{VLM}, i}^{(L24)}) \big\|_2 \, \big\| \mathbf{z}_{\text{pooled}, i} \big\|_2}$$

---

## 4. Module 2: Dual-Head World-Model Predictor

The world model forecasts how the environment evolves conditioned on robot action intentions.

```text
 ┌────────────────────────────────────────────────────────┐
 │ Context Visual Latents (s_t)   Action Tokens (h_act)   │
 └───────────────────┬──────────────────────┬─────────────┘
                     │                      │
                     ▼                      ▼
 ┌────────────────────────────────────────────────────────┐
 │     Action-Conditioned Transformer Predictor (RoPE)    │
 └──────────────────────────┬─────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
 ┌─────────────────────────┐ ┌─────────────────────────┐
 │ Head 1: Semantic State  │ │ Head 2: Geometric Dynamics│
 │ ŝ_sem in R^(T x S x D)  │ │ ŝ_geo in R^(T x 64 x 2) │
 └────────────┬────────────┘ └────────────┬────────────┘
              ▼                           ▼
          L_WM_sem                    L_WM_geo
    (L1 vs Frozen V-JEPA2)      (SmoothL1 vs 3D Tracks)
```

### 4.1 Semantic Future State Head ($\hat{\mathbf{s}}^{\text{sem}}$)
Forecasts high-level video latent representations supervised by the frozen **V-JEPA2** encoder:

$$\mathcal{L}_{\text{WM}}^{\text{sem}} = \frac{1}{T \cdot S \cdot D} \sum_{t=1}^T \sum_{s=1}^S \Big\| \hat{\mathbf{s}}_{t, s}^{\text{sem}} - \text{stop-gradient}(\mathbf{s}_{t, s}^{\text{V-JEPA2}}) \Big\|_1$$

### 4.2 Geometric Dynamics Head ($\hat{\mathbf{s}}^{\text{geo}}$)
Forecasts 2D/3D point displacement trajectories for $N_p=64$ query points across future horizon $T$:

$$\mathcal{L}_{\text{WM}}^{\text{geo}} = \frac{1}{T \cdot N_p} \sum_{t=1}^T \sum_{p=1}^{N_p} \text{SmoothL1}\Big( \Delta \hat{\mathbf{p}}_{t, p}, \, \text{stop-gradient}(\Delta \mathbf{p}_{t, p}^{\text{canon}}) \Big)$$

where $\text{SmoothL1}(x)$ is defined as:

$$\text{SmoothL1}(x) = \begin{cases} 0.5 x^2 & \text{if } |x| < 1.0 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

---

## 5. Module 3: Temporal Coordinate Canonicalization

To ensure that geometric predictions are invariant to camera motion (panning, tilting, or height adjustments), raw metric coordinates are canonicalized to **anchor frame $t=0$**.

### 5.1 Depth Unprojection to Camera Space
Given metric depth map $Z(u, v) \in \mathbb{R}^{H \times W}$ and camera intrinsics matrix $K$:

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

Each pixel $(u, v)$ is unprojected into Euclidean 3D camera coordinates:

$$\mathbf{X}_{\text{cam}}(u, v) = Z(u, v) \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} \frac{(u - c_x) \cdot Z(u, v)}{f_x} \\ \frac{(v - c_y) \cdot Z(u, v)}{f_y} \\ Z(u, v) \end{bmatrix}$$

### 5.2 Rigid World-to-Anchor Transformation
Using VGGT's predicted 9D camera rotation matrices $R_t \in \text{SO}(3)$ and translation vectors $\mathbf{t}_t \in \mathbb{R}^3$:

$$\mathbf{X}_{\text{world}}(t) = R_t^\top \big( \mathbf{X}_{\text{cam}}(t) - \mathbf{t}_t \big)$$

All future points $\mathbf{X}_{\text{world}}(t)$ are transformed into the fixed reference coordinate frame of frame 0:

$$\mathbf{X}_{\text{canon}}(t) = R_0 \, \mathbf{X}_{\text{world}}(t) + \mathbf{t}_0$$

### 5.3 Canonical Point-Track Displacement Targets
The target displacement vectors for query track $p$ between step $t$ and $t+1$ are:

$$\Delta \mathbf{p}_{t, p}^{\text{canon}} = \mathbf{p}_{t+1, p}^{\text{canon}} - \mathbf{p}_{t, p}^{\text{canon}}$$

---

## 6. Module 4: Flow-Matching Action Head

Geo-JEPA implements a **Conditional Vector Field Flow-Matching** architecture (Diffusion Transformer DiT-B) to predict continuous action chunks.

### 6.1 Action Chunk Parameterization
The policy outputs an 8-step continuous action chunk:

$$\mathbf{A} = [\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_H] \in \mathbb{R}^{H \times 7}, \quad H = 8$$

where each action step contains:

$$\mathbf{a}_t = \big[ \Delta x_t, \, \Delta y_t, \, \Delta z_t, \, \Delta \text{roll}_t, \, \Delta \text{pitch}_t, \, \Delta \text{yaw}_t, \, \text{grip}_t \big]$$

### 6.2 Flow-Matching Vector Field Regression
Given target trajectory $\mathbf{x}_1 = \mathbf{A}$, noise sample $\mathbf{x}_0 \sim \mathcal{N}(0, \mathbf{I})$, and interpolation timestep $t \in [0, 1]$:

$$\mathbf{x}_t = (1 - t) \mathbf{x}_0 + t \mathbf{x}_1$$

The flow-matching vector field network $v_\theta(\mathbf{x}_t, t, \mathbf{c})$ is trained by minimizing:

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \Big\| v_\theta(\mathbf{x}_t, t, \mathbf{c}) - (\mathbf{x}_1 - \mathbf{x}_0) \Big\|_2^2$$

where $\mathbf{c} = \mathbf{h}_{\text{embodied}}$ is the conditioning hidden state from the VLM.

---

## 7. Module 5: Action 3D Rays & MP-Geo Guidance

### 7.1 Action-Grounded 3D Ray Bundles ($\mathcal{L}_{\text{ray}}$)
Supervises latent action tokens to predict unit line-of-sight reach rays $\hat{\mathbf{r}} \in \mathbb{R}^3$ and metric reach distance $\hat{d} \in \mathbb{R}^1$:

$$\mathcal{L}_{\text{ray}} = \left( 1 - \frac{\hat{\mathbf{r}} \cdot \mathbf{r}_{\text{gt}}}{\|\hat{\mathbf{r}}\|_2 \, \|\mathbf{r}_{\text{gt}}\|_2} \right) + 0.5 \cdot \text{SmoothL1}(\hat{d}, d_{\text{gt}})$$

where $\mathbf{r}_{\text{gt}} = \mathbf{X}_{\text{target}}^{\text{3D}} - \mathbf{X}_{\text{gripper}}^{\text{3D}}$ and $d_{\text{gt}} = \|\mathbf{r}_{\text{gt}}\|_2$.

### 7.2 Model-Predictive Geometric Guidance (`MP-Geo Guidance`)
At test time, the policy samples $K=8$ candidate action chunks $\mathbf{a}^{(1 \dots K)}$, predicts future point tracks via $\hat{s}^{\text{geo}}$, and scores them:

$$J(\mathbf{a}) = w_{\text{target}} J_{\text{target}}(\mathbf{a}) + w_{\text{smooth}} J_{\text{smooth}}(\mathbf{a}) + w_{\text{clearance}} J_{\text{clearance}}(\mathbf{a})$$

- **$J_{\text{target}}(\mathbf{a}) = -\|\hat{\mathbf{p}}_{t+T}(\mathbf{a}) - \mathbf{X}_{\text{target}}\|_2$**: Minimizes 3D distance to target.
- **$J_{\text{smooth}}(\mathbf{a}) = -\|\Delta^2 \mathbf{a}\|_2$**: Penalizes second-order acceleration jitter.
- **$J_{\text{clearance}}(\mathbf{a}) = -\text{ReLU}(-\Delta z - 0.4)$**: Prevents table collision.

$$\mathbf{a}^* = \arg\max_{i \in [1, K]} J(\mathbf{a}^{(i)})$$

---

## 8. Dynamic Loss Scheduling

To prevent representation collapse during early training, $\alpha(t)$ and $\gamma(t)$ follow a progressive linear warmup:

$$\alpha(t) = \min\left( \alpha_{\text{target}}, \, \alpha_{\text{target}} \cdot \frac{t}{T_{\text{warmup}}} \right), \quad \alpha_{\text{target}} = 0.50, \quad T_{\text{warmup}} = 1000$$

$$\gamma(t) = \min\left( \gamma_{\text{target}}, \, \gamma_{\text{target}} \cdot \frac{t}{T_{\text{warmup}}} \right), \quad \gamma_{\text{target}} = 0.10, \quad T_{\text{warmup}} = 1000$$

$$\beta = 0.10 \quad (\text{constant})$$

---

## 9. Tensor Dimensions & Shape Trace Table

| Tensor Variable | Shape | Description |
| :--- | :--- | :--- |
| **`image_agent`** | $(B, 3, 224, 224)$ | AgentView RGB third-person camera |
| **`image_wrist`** | $(B, 3, 224, 224)$ | WristView RGB end-effector camera |
| **`input_ids`** | $(B, S_{\text{seq}} = 128)$ | Tokenized prompt sequence |
| **`vlm_hidden_l24`** | $(B, S_{\text{vis}} = 32, 1024)$ | Layer 24 visual patch tokens for alignment |
| **`vggt_backbone_latents`** | $(B, V=2, 1369, 2048)$ | Pre-cached VGGT Layer 24 teacher latents |
| **`vggt_pooled`** | $(B, S_{\text{vis}} = 32, 2048)$ | Interpolate-pooled VGGT features with UV PE |
| **`action_tokens`** | $(B, N_{\text{act}} = 9, 1024)$ | Latent action conditioning tokens |
| **`pred_sem` ($\hat{s}^{\text{sem}}$)** | $(B, 48, 1024)$ | Predicted semantic future states |
| **`gt_sem` ($s^{\text{V-JEPA2}}$)** | $(B, 48, 1024)$ | Frozen V-JEPA2 target states |
| **`pred_geo` ($\hat{s}^{\text{geo}}$)** | $(B, 48, 128)$ | Predicted 3D point-track displacements |
| **`gt_geo` ($\Delta \mathbf{p}^{\text{canon}}$)** | $(B, 48, 128)$ | Canonicalized ground-truth 3D point tracks |
| **`embodied_token`** | $(B, 1024)$ | Final hidden state conditioning action head |
| **`predicted_action_chunk`** | $(B, H=8, D_{\text{act}}=7)$ | Continuous 7-DoF robot trajectory chunk |

---

*Geo-JEPA Architecture Reference — Team M3OW — [GitHub Repository](https://github.com/Team-M3OW/Geo-JEPA)*
