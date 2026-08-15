"""
LIBERO Policy Wrapper for Geo-JEPA Evaluation.

Handles:
- Processing real-time environment observations (agentview + wristview RGB)
- Formatting language instructions
- Querying Geo-JEPA policy to predict 7-DoF action chunks
- Temporal action ensembling and receding horizon execution
"""

from typing import Dict, List, Optional, Tuple, Union
import collections
import numpy as np
import torch
from PIL import Image


class GeoJEPALiberoPolicy:
    """
    Inference policy interface for LIBERO benchmark environments.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        action_horizon: int = 8,
        image_size: int = 256,
        receding_horizon_steps: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.receding_horizon = receding_horizon_steps
        self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()

        # Action chunk buffer for receding horizon execution
        self.action_buffer = collections.deque()
        self.current_instruction = None

    def reset(self, instruction: Optional[str] = None):
        """Reset policy action buffers for a new evaluation episode."""
        self.action_buffer.clear()
        self.current_instruction = instruction

    def set_instruction(self, instruction: str):
        self.current_instruction = instruction

    def process_observation(self, obs: Dict[str, np.ndarray]) -> List[Image.Image]:
        """
        Extract dual camera views from LIBERO observation dict.
        """
        # Primary agentview
        if "agentview_image" in obs:
            raw_agent = obs["agentview_image"]
        elif "agentview_rgb" in obs:
            raw_agent = obs["agentview_rgb"]
        else:
            raw_agent = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Eye-in-hand wristview
        if "robot0_eye_in_hand_image" in obs:
            raw_wrist = obs["robot0_eye_in_hand_image"]
        elif "eye_in_hand_rgb" in obs:
            raw_wrist = obs["eye_in_hand_rgb"]
        else:
            raw_wrist = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # LIBERO renders upside down in some OpenGL backends; handle flipping if needed
        # Convert to PIL and resize
        img_agent = Image.fromarray(raw_agent).resize((self.image_size, self.image_size), Image.BILINEAR)
        img_wrist = Image.fromarray(raw_wrist).resize((self.image_size, self.image_size), Image.BILINEAR)

        return [img_agent, img_wrist]

    @torch.no_grad()
    def step(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Execute single policy step. If action buffer is non-empty, returns next buffered action;
        otherwise queries model to predict a new chunk.
        """
        if len(self.action_buffer) > 0:
            return self.action_buffer.popleft()

        # Format input images and state
        images = self.process_observation(obs)
        instruction = self.current_instruction if self.current_instruction is not None else "manipulate object"
        
        # Proprioceptive state
        state = None
        if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
            pos = obs["robot0_eef_pos"]
            quat = obs["robot0_eef_quat"]
            gripper = obs.get("robot0_gripper_qpos", np.zeros(2))
            state = np.concatenate([pos, quat, gripper], axis=-1)
        elif "ee_states" in obs:
            state = obs["ee_states"]

        # Forward pass through model
        if hasattr(self.model, "predict_action"):
            pred_chunk = self.model.predict_action(
                batch_images=[images],
                instructions=[instruction],
                state=state[None] if state is not None else None
            )
        else:
            # Fallback for mock/generic model
            pred_chunk = np.zeros((1, self.action_horizon, 7), dtype=np.float32)

        if isinstance(pred_chunk, torch.Tensor):
            pred_chunk = pred_chunk.detach().cpu().numpy()

        actions = pred_chunk[0]  # (horizon, 7)

        # Post-process gripper action (binarize threshold)
        for act in actions[:self.receding_horizon]:
            act_copy = act.copy()
            # Binarize gripper dimension (index 6): 1.0 open, -1.0 close
            if len(act_copy) >= 7:
                act_copy[6] = 1.0 if act_copy[6] > 0.0 else -1.0
            self.action_buffer.append(act_copy)

        return self.action_buffer.popleft()
