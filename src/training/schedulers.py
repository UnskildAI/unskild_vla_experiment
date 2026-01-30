"""LR schedulers for training (cosine, linear, constant)."""

from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ConstantLR, LRScheduler
from pydantic import BaseModel, Field


class SchedulerConfig(BaseModel):
    """LR scheduler config."""

    name: str = "cosine"
    num_training_steps: int = Field(10_000, gt=0)
    warmup_steps: int = Field(0, ge=0)
    extra: dict[str, Any] = Field(default_factory=dict)


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig | dict[str, Any],
    **kwargs: Any,
) -> LRScheduler:
    """Build LR scheduler from config."""
    if isinstance(config, dict):
        config = SchedulerConfig(**config)
    name = config.name.lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.num_training_steps, **kwargs)
    if name == "linear":
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=config.num_training_steps, **kwargs)
    if name == "constant":
        return ConstantLR(optimizer, factor=1.0, **kwargs)
    raise ValueError(f"Unknown scheduler: {name}")
