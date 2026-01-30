"""Action loss: L2 / MSE or custom for vision → action."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from src.core.typing import TensorDict


class ActionLossConfig(BaseModel):
    """Config for action loss."""

    loss_type: str = Field("mse", description="mse, l1, huber")
    reduction: str = Field("mean")
    extra: dict[str, Any] = Field(default_factory=dict)


class ActionLoss(nn.Module):
    """Action regression loss: MSE / L1 / Huber. Dataset-agnostic supervision."""

    def __init__(self, config: ActionLossConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.config = config or ActionLossConfig(**kwargs)
        if self.config.loss_type == "mse":
            self._loss = nn.MSELoss(reduction=self.config.reduction)
        elif self.config.loss_type == "l1":
            self._loss = nn.L1Loss(reduction=self.config.reduction)
        else:
            self._loss = nn.HuberLoss(reduction=self.config.reduction)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if mask is not None:
            pred = pred * mask
            target = target * mask
        return self._loss(pred, target)
