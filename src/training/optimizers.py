"""Optimizer builders from config (AdamW, etc.)."""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import AdamW, Optimizer
from pydantic import BaseModel, Field


class OptimizerConfig(BaseModel):
    """Optimizer config (AdamW)."""

    name: str = "adamw"
    lr: float = Field(1e-5, gt=0)
    weight_decay: float = Field(0.0, ge=0)
    betas: tuple[float, float] = (0.9, 0.999)
    extra: dict[str, Any] = Field(default_factory=dict)


def build_optimizer(
    parameters: Any,
    config: OptimizerConfig | dict[str, Any],
    **kwargs: Any,
) -> Optimizer:
    """Build optimizer from config. No hard-coded model assumptions."""
    if isinstance(config, dict):
        config = OptimizerConfig(**config)
    name = config.name.lower()
    if name == "adamw":
        return AdamW(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            **kwargs,
        )
    raise ValueError(f"Unknown optimizer: {name}")
