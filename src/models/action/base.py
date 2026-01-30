"""BaseActionHead: ABC + torch-native, Pydantic config, checkpoint. Separable from VLM/diffusion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel

from src.core.typing import TensorDict


class BaseActionHeadConfig(BaseModel):
    """Base config for action heads."""

    input_dim: int = 0
    action_dim: int = 0
    extra: dict[str, Any] = {}


class BaseActionHead(nn.Module, ABC):
    """Abstract base action head. Torch-native, config-driven, checkpointable. Can be frozen/swapped."""

    def __init__(self, config: BaseActionHeadConfig, **kwargs: Any) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, features: torch.Tensor, conditioning: TensorDict | None = None, **kwargs: Any) -> torch.Tensor:
        """Map features (e.g. vision embeddings) to action. conditioning optional."""
        pass

    def get_config(self) -> BaseActionHeadConfig:
        return self.config
