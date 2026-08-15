"""
LIBERO Dataset Loader for Geo-JEPA Fine-Tuning & Evaluation.

Loads:
- Dual camera views: agentview (third-person) + eye-in-hand (wrist camera)
- Proprioceptive states (EEF pose / joint states)
- Future continuous action chunks (7-DoF: dx, dy, dz, droll, dpitch, dyaw, gripper)
- Natural language task instructions
- Precached VGGT geometric features (.npz) for fast geometric alignment & temporal loss computation
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LiberoHDF5Dataset(Dataset):
    """
    Dataset class for loading LIBERO demonstrations from standard HDF5 archives.
    """

    def __init__(
        self,
        hdf5_file_path: Union[str, Path],
        action_horizon: int = 8,
        future_action_window_size: int = 7,
        past_action_window_size: int = 0,
        img_size: int = 256,
        cache_dir: Optional[Union[str, Path]] = None,
        task_instruction: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.hdf5_file_path = str(hdf5_file_path)
        self.action_horizon = action_horizon
        self.future_window = future_action_window_size
        self.past_window = past_action_window_size
        self.chunk_len = self.past_window + 1 + self.future_window
        self.img_size = img_size
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.default_instruction = task_instruction

        # Scan demonstration index boundaries
        self.demo_keys = []
        self.index_map = []  # List of (demo_key, timestep_idx)

        with h5py.File(self.hdf5_file_path, "r") as f:
            data_group = f["data"]
            self.demo_keys = sorted(list(data_group.keys()))
            
            # Extract language instruction if stored in attributes or root
            if self.default_instruction is None:
                if "problem_info" in f.attrs:
                    self.default_instruction = str(f.attrs.get("problem_info", ""))
                else:
                    stem = Path(self.hdf5_file_path).stem
                    self.default_instruction = stem.replace("_demo", "").replace("_", " ")

            for d_key in self.demo_keys:
                demo = data_group[d_key]
                num_steps = demo["actions"].shape[0]
                for t in range(num_steps):
                    self.index_map.append((d_key, t, num_steps))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Union[List[Image.Image], np.ndarray, str]]:
        demo_key, t, num_steps = self.index_map[idx]

        with h5py.File(self.hdf5_file_path, "r") as f:
            demo = f["data"][demo_key]
            obs = demo["obs"]

            # 1. Dual camera views
            agentview_raw = obs["agentview_rgb"][t]          # (H, W, 3) uint8
            wristview_raw = obs["eye_in_hand_rgb"][t]        # (H, W, 3) uint8

            # Invert BGR/RGB if needed and convert to PIL
            if agentview_raw.ndim == 3 and agentview_raw.shape[-1] == 3:
                # LIBERO stores images as RGB uint8 (typically 128x128 or 256x256)
                img_agent = Image.fromarray(agentview_raw).resize((self.img_size, self.img_size), Image.BILINEAR)
                img_wrist = Image.fromarray(wristview_raw).resize((self.img_size, self.img_size), Image.BILINEAR)
            else:
                img_agent = Image.new("RGB", (self.img_size, self.img_size))
                img_wrist = Image.new("RGB", (self.img_size, self.img_size))

            # 2. EEF / Proprioceptive State
            if "ee_states" in obs:
                state = obs["ee_states"][t]  # (6,) or (8,)
            elif "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
                pos = obs["robot0_eef_pos"][t]
                quat = obs["robot0_eef_quat"][t]
                gripper = obs["robot0_gripper_qpos"][t] if "robot0_gripper_qpos" in obs else np.zeros(2)
                state = np.concatenate([pos, quat, gripper], axis=-1)
            else:
                state = np.zeros(7, dtype=np.float32)

            # 3. Future Action Chunk Extraction
            all_actions = demo["actions"][:]  # (T_demo, action_dim)
            start_t = max(0, t - self.past_window)
            end_t = min(num_steps, t + self.future_window + 1)
            
            action_slice = all_actions[start_t:end_t]
            # Pad if slice is shorter than chunk_len
            if len(action_slice) < self.chunk_len:
                pad_len = self.chunk_len - len(action_slice)
                last_act = action_slice[-1:] if len(action_slice) > 0 else np.zeros((1, 7))
                padding = np.repeat(last_act, pad_len, axis=0)
                action_slice = np.concatenate([action_slice, padding], axis=0)

        # 4. Optional Precached VGGT features
        vggt_curr_latent = None
        vggt_geo_target = None
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"{demo_key}_t{t:04d}.npz"
            if cache_file.exists():
                cached = np.load(cache_file)
                if "backbone_latents" in cached:
                    vggt_curr_latent = cached["backbone_latents"]
                if "track_displacements" in cached:
                    vggt_geo_target = cached["track_displacements"]

        item = {
            "image": [img_agent, img_wrist],
            "video": np.stack([np.array(img_agent), np.array(img_wrist)])[None],  # [1, 2, H, W, 3]
            "state": state.astype(np.float32),
            "action": action_slice.astype(np.float32),
            "lang": self.default_instruction,
        }

        if vggt_curr_latent is not None:
            item["vggt_current_latent"] = vggt_curr_latent
        if vggt_geo_target is not None:
            item["vggt_geo_target"] = vggt_geo_target

        return item


def libero_collate_fn(batch: List[dict]) -> List[dict]:
    """Collate function returning standard Geo-JEPA / VLA-JEPA dictionary format."""
    return batch
