"""
Warmup Schedulers for Geo-JEPA Loss Coefficients.

Provides progressive scaling for geometric alignment weight (alpha) and
geometric world model prediction weight (gamma), ensuring stable visual
representation formation before joint optimization.
"""

import math
from typing import Dict, Optional, Union


class CoefficientWarmupScheduler:
    """
    Schedules loss weights (e.g. alpha for L_geo, gamma for L_WM_geo) over training steps.
    """

    def __init__(
        self,
        target_val: float = 0.5,
        warmup_steps: int = 2000,
        schedule_type: str = "linear",
        start_val: float = 0.0,
    ) -> None:
        """
        Args:
            target_val: Final target coefficient value (e.g. 0.5 for alpha)
            warmup_steps: Number of optimization steps over which to ramp up
            schedule_type: 'linear', 'cosine', or 'constant'
            start_val: Starting value at step 0 (default: 0.0)
        """
        self.target_val = float(target_val)
        self.warmup_steps = int(warmup_steps)
        self.schedule_type = schedule_type.lower()
        self.start_val = float(start_val)

    def get_val(self, step: int) -> float:
        """
        Compute the scheduled coefficient at the given global step.
        """
        if self.warmup_steps <= 0 or self.schedule_type == "constant":
            return self.target_val

        if step >= self.warmup_steps:
            return self.target_val

        progress = max(0.0, float(step) / float(self.warmup_steps))

        if self.schedule_type == "linear":
            return self.start_val + (self.target_val - self.start_val) * progress
        elif self.schedule_type == "cosine":
            factor = 0.5 * (1.0 - math.cos(math.pi * progress))
            return self.start_val + (self.target_val - self.start_val) * factor
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")


class GeoJEPALossScheduler:
    """
    Unified manager for all loss weights:
    - alpha: Current-frame geometric alignment loss weight (with warmup)
    - beta:  Semantic world model loss weight
    - gamma: Geometric world model prediction loss weight (with warmup)
    """

    def __init__(
        self,
        alpha_target: float = 0.5,
        alpha_warmup_steps: int = 2000,
        alpha_schedule: str = "linear",
        beta: float = 0.1,
        gamma_target: float = 0.1,
        gamma_warmup_steps: int = 2000,
        gamma_schedule: str = "linear",
    ) -> None:
        self.alpha_scheduler = CoefficientWarmupScheduler(
            target_val=alpha_target,
            warmup_steps=alpha_warmup_steps,
            schedule_type=alpha_schedule
        )
        self.gamma_scheduler = CoefficientWarmupScheduler(
            target_val=gamma_target,
            warmup_steps=gamma_warmup_steps,
            schedule_type=gamma_schedule
        )
        self.beta = beta

    def get_coefficients(self, step: int) -> Dict[str, float]:
        """
        Return dict of active loss multipliers at current step.
        """
        return {
            "alpha": self.alpha_scheduler.get_val(step),
            "beta": self.beta,
            "gamma": self.gamma_scheduler.get_val(step),
        }
