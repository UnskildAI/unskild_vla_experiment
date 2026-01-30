"""BaseDiffusionModel: ABC + torch-native, Pydantic config, checkpoint, continuous time."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel

from src.core.typing import TensorDict


class BaseDiffusionConfig(BaseModel):
    """Base config for diffusion / flow-matching models."""

    in_channels: int = 3
    out_channels: int = 3
    conditioning_dim: int = 0
    extra: dict[str, Any] = {}


class BaseDiffusionModel(nn.Module, ABC):
    """Abstract base diffusion model. Torch-native, config-driven, checkpointable, continuous time."""

    def __init__(self, config: BaseDiffusionConfig, **kwargs: Any) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: TensorDict | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict velocity / noise. x: noisy data, t: time in [0,1], conditioning optional."""
        pass

    def get_config(self) -> BaseDiffusionConfig:
        return self.config
