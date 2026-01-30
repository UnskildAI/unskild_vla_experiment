"""Flow-matching / diffusion loss: explicit noise schedules, continuous time, pluggable."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from src.core.typing import TensorDict


class FlowMatchingLossConfig(BaseModel):
    """Config for flow-matching loss."""

    reduction: str = Field("mean", description="mean, sum, none")
    extra: dict[str, Any] = Field(default_factory=dict)


class FlowMatchingLoss(nn.Module):
    """Flow-matching loss: L2(predicted_velocity - target_velocity). Continuous time, pluggable."""

    def __init__(self, config: FlowMatchingLossConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.config = config or FlowMatchingLossConfig(**kwargs)
        self.reduction = self.config.reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        diff = (pred - target) ** 2
        if mask is not None:
            diff = diff * mask
        if self.reduction == "mean":
            return diff.mean()
        if self.reduction == "sum":
            return diff.sum()
        return diff
