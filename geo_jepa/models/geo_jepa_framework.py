"""
Geo-JEPA Framework.

Implements the unified Geometric-Temporal Joint Embedding Predictive Architecture:
- Backbone: Qwen-VL with action token injection
- Action Head: Flow-Matching Action Head (L_FM)
- Mid-Depth Geometric Alignment: Spatial-Forcing AlignProjector against frozen VGGT (alpha * L_geo)
- Temporal Dynamics: Dual-head leakage-free World Model predicting semantic + geometric future states (beta * L_WM_sem + gamma * L_WM_geo)

Total Loss:
  L = L_FM + beta * L_WM_sem + gamma * L_WM_geo + alpha * L_geo
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor

import sys
sys.path.insert(0, "/home/kavinder/geo-jepa-dev/VLA-JEPA")
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

from geo_jepa.models.qwen_alignment_hook import QwenGeometricAlignmentHook
from geo_jepa.models.dual_head_predictor import DualHeadVisionTransformerPredictor

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("Geo_JEPA")
class Geo_JEPA(baseframework):
    """
    Geo-JEPA Framework combining geometric grounding with leakage-free temporal dynamics.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.config = config

        # 1. Initialize Qwen-VL Interface
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        embodied_action_token = self.config.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token
        )

        # 2. Action Flow Matching Head
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # 3. Frozen V-JEPA2 Semantic Target Encoder
        self.vj_encoder = AutoModel.from_pretrained(self.config.framework.vj2_model.base_encoder)
        self.vj_processor = AutoVideoProcessor.from_pretrained(self.config.framework.vj2_model.base_encoder)
        self.vj_encoder.eval()
        for param in self.vj_encoder.parameters():
            param.requires_grad = False

        tubelet_size = self.vj_encoder.config.tubelet_size
        
        # 4. Geometric Alignment Hook (Phase 2)
        geo_config = getattr(config.framework, "geometric_forcing", {})
        self.enable_geo_alignment = geo_config.get("enable_geo_alignment", True)
        self.alignment_layer_idx = geo_config.get("alignment_layer_idx", 24)  # Default ~75% depth
        self.alpha_default = geo_config.get("alpha", 0.5)

        self.geo_hook = QwenGeometricAlignmentHook(
            vlm_dim=self.qwen_vl_interface.model.config.hidden_size,
            vggt_dim=1024,
            alignment_layer_idx=self.alignment_layer_idx,
            image_token_id=151655
        )

        # 5. Dual-Head World Model Predictor (Phase 3)
        self.enable_geo_wm_head = geo_config.get("enable_geo_wm_head", True)
        self.beta = config.framework.vj2_model.get("beta", 0.1)
        self.gamma_default = geo_config.get("gamma", 0.1)
        geo_target_dim = geo_config.get("geo_target_dim", 128)  # 64 tracks * 2 coords = 128

        self.dual_predictor = DualHeadVisionTransformerPredictor(
            img_size=((self.vj_encoder.config.image_size, self.vj_encoder.config.image_size)),
            tubelet_size=1,
            num_frames=self.config.framework.vj2_model.num_frames // tubelet_size,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim_semantic=self.vj_encoder.config.hidden_size * 2,  # multi view
            geo_target_dim=geo_target_dim,
            action_embed_dim=self.qwen_vl_interface.model.config.hidden_size,
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
        )

        # Prompts
        self.replace_prompt = "".join(
            [each * self.config.framework.vj2_model.num_action_tokens_per_timestep for each in
             action_tokens[:self.config.framework.vj2_model.num_frames // tubelet_size - 1]]
        )
        self.embodied_replace_prompt = "".join(
            [embodied_action_token * self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction]
        )

    def expand_tokenizer(
        self,
        tokenizer,
        special_action_token: str = "<|action_{}|>",
        max_action_tokens: int = 32,
        embodied_action_token: str = "<|embodied_action|>"
    ):
        action_tokens, action_token_ids = [], []
        for i in range(0, max_action_tokens):
            token_i = special_action_token.format(i)
            action_tokens.append(token_i)
            if token_i not in tokenizer.get_vocab():
                tokenizer.add_tokens([token_i], special_tokens=True)
            action_token_ids.append(tokenizer.convert_tokens_to_ids(token_i))
            
        if embodied_action_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([embodied_action_token], special_tokens=True)
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
            
        return action_tokens, action_token_ids, embodied_action_token_id

    def forward(
        self,
        examples: List[dict] = None,
        alpha: Optional[float] = None,
        gamma: Optional[float] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing:
          L_total = L_FM + beta * L_WM_sem + gamma * L_WM_geo + alpha * L_geo
        """
        active_alpha = alpha if alpha is not None else self.alpha_default
        active_gamma = gamma if gamma is not None else self.gamma_default

        batch_images = [example["image"] for example in examples]
        batch_videos = [example["video"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples] if "action" in examples[0] else None
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        # Precached geometric features (from Phase 1 caching)
        vggt_curr_latents = [example["vggt_current_latent"] for example in examples] if "vggt_current_latent" in examples[0] else None
        vggt_geo_targets = [example["vggt_geo_target"] for example in examples] if "vggt_geo_target" in examples[0] else None

        batch_videos = np.stack(batch_videos).transpose(0, 1, 2, 5, 3, 4)  # [B, V, T, 3, H, W]

        # Step 1: Qwen-VL Forward
        if actions is not None:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                prompt_replace_dict={"{actions}": self.replace_prompt, "{e_actions}": self.embodied_replace_prompt},
                prompt_template=self.config.datasets.vla_data.get("CoT_prompt", "")
            )
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                prompt_replace_dict={"{actions}": self.replace_prompt},
                prompt_template=self.config.datasets.video_data.get("CoT_prompt", "")
            )

        action_indices = torch.isin(qwen_inputs["input_ids"], torch.tensor(self.action_token_ids, device=qwen_inputs["input_ids"].device)).nonzero(as_tuple=True)
        embodied_action_indices = torch.isin(qwen_inputs["input_ids"], torch.tensor([self.embodied_action_token_id], device=qwen_inputs["input_ids"].device)).nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            B, _, H = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)

            # Step 2: Geometric Alignment Loss (Phase 2)
            geo_align_loss = torch.tensor(0.0, device=last_hidden.device)
            if self.enable_geo_alignment and vggt_curr_latents is not None:
                vggt_curr_tensor = torch.tensor(np.stack(vggt_curr_latents), device=last_hidden.device, dtype=torch.bfloat16)
                geo_align_loss = self.geo_hook.compute_geometric_loss(
                    hidden_states=qwenvl_outputs.hidden_states,
                    input_ids=qwen_inputs["input_ids"],
                    vggt_current_features=vggt_curr_tensor
                )

            # Step 3: V-JEPA2 Semantic Feature Extraction (Leakage-Free, Stop Gradient)
            B, V, T, C, H_img, W_img = batch_videos.shape
            flat_videos = batch_videos.reshape(B * V, T, C, H_img, W_img)
            input_videos = []
            for i in range(B * V):
                input_videos.append(self.vj_processor(videos=flat_videos[i], return_tensors="pt")["pixel_values_videos"].to(self.vj_encoder.device))
            input_videos = torch.cat(input_videos, dim=0)

            with torch.no_grad():
                video_embeddings = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
                video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=V, dim=0), dim=2)

            T_step = T // self.vj_encoder.config.tubelet_size
            input_states = video_embeddings[:, :video_embeddings.shape[1] // T_step * (T_step - 1), :]
            gt_sem_states = video_embeddings[:, video_embeddings.shape[1] // T_step:, :]

            # Step 4: Dual-Head World Model Predictor Forward (Phase 3)
            pred_sem_states, pred_geo_states = self.dual_predictor(input_states, action_tokens)

            # Geometric future target (stop gradient)
            if vggt_geo_targets is not None:
                gt_geo_tensor = torch.tensor(np.stack(vggt_geo_targets), device=last_hidden.device, dtype=torch.bfloat16)
            else:
                # Fallback mock target
                gt_geo_tensor = pred_geo_states.detach()

            wm_losses = self.dual_predictor.compute_dual_wm_loss(
                pred_sem_states=pred_sem_states,
                gt_sem_states=gt_sem_states,
                pred_geo_states=pred_geo_states,
                gt_geo_states=gt_geo_tensor,
                gamma=active_gamma
            )

        if "action" not in examples[0]:
            # Human video pretraining mode
            return {
                "wm_loss_sem": wm_losses["wm_loss_sem"] * self.beta,
                "wm_loss_geo": wm_losses["wm_loss_geo"] * active_gamma,
                "geo_align_loss": geo_align_loss * active_alpha,
            }

        # Step 5: Flow Matching Action Head Forward
        with torch.autocast("cuda", dtype=torch.float32):
            actions_t = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions_t[:, -(self.future_action_window_size + 1):, :]

            repeated_steps = self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            actions_target_rep = actions_target.repeat(repeated_steps, 1, 1)
            embodied_rep = embodied_action_tokens.repeat(repeated_steps, 1, 1)

            state_rep = None
            if state is not None:
                state_t = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_rep = state_t.repeat(repeated_steps, 1, 1)

            action_loss = self.action_model(embodied_rep, actions_target_rep, state_rep)

        return {
            "action_loss": action_loss,
            "wm_loss_sem": wm_losses["wm_loss_sem"] * self.beta,
            "wm_loss_geo": wm_losses["wm_loss_geo"] * active_gamma,
            "geo_align_loss": geo_align_loss * active_alpha,
        }
