"""MLP and diffusion-based action heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from src.models.action.base import BaseActionHead, BaseActionHeadConfig
from src.core.typing import TensorDict


class MLPActionHeadConfig(BaseActionHeadConfig):
    """Config for MLP action head."""

    input_dim: int = Field(..., gt=0)
    action_dim: int = Field(..., gt=0)
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 256])
    extra: dict[str, Any] = Field(default_factory=dict)


class MLPActionHead(BaseActionHead):
    """MLP policy: features -> action. Separable, freezeable."""

    def __init__(self, config: MLPActionHeadConfig | dict[str, Any], **kwargs: Any) -> None:
        if isinstance(config, dict):
            config = MLPActionHeadConfig(**config)
        super().__init__(config)
        dims = [config.input_dim] + list(config.hidden_dims) + [config.action_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self._mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, conditioning: TensorDict | None = None, **kwargs: Any) -> torch.Tensor:
        return self._mlp(features)


class DiffusionActionHeadConfig(BaseActionHeadConfig):
    """Config for diffusion-based action head (e.g. action chunk generation)."""

    input_dim: int = Field(..., gt=0)
    action_dim: int = Field(..., gt=0)
    chunk_length: int = Field(16, gt=0)
    extra: dict[str, Any] = Field(default_factory=dict)


class DiffusionActionHead(BaseActionHead):
    """Diffusion over action sequence. Stub: conditions on features, outputs action chunk."""

    def __init__(self, config: DiffusionActionHeadConfig | dict[str, Any], **kwargs: Any) -> None:
        if isinstance(config, dict):
            config = DiffusionActionHeadConfig(**config)
        super().__init__(config)
        self.chunk_length = config.chunk_length
        self._proj = nn.Linear(config.input_dim, config.action_dim * config.chunk_length)

    def forward(self, features: torch.Tensor, conditioning: TensorDict | None = None, **kwargs: Any) -> torch.Tensor:
        out = self._proj(features)
        return out.view(features.shape[0], self.chunk_length, -1)


class VisionActionWrapperConfig(BaseModel):
    """Config for vision -> action wrapper."""

    input_dim: int = Field(2048, gt=0)
    action_dim: int = Field(7, gt=0)
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 256])
    extra: dict[str, Any] = Field(default_factory=dict)


class VisionActionWrapper(nn.Module):
    """Wrapper: pixel_values -> pooled features -> MLPActionHead. For pipelines that pass pixel_values."""

    def __init__(self, config: VisionActionWrapperConfig | MLPActionHeadConfig | dict[str, Any], **kwargs: Any) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = VisionActionWrapperConfig(**config)
        elif isinstance(config, MLPActionHeadConfig):
            config = VisionActionWrapperConfig(input_dim=config.input_dim, action_dim=config.action_dim, hidden_dims=getattr(config, "hidden_dims", [256, 256]))
        self.config = config
        self._pool = nn.AdaptiveAvgPool2d(1)
        head_config = MLPActionHeadConfig(input_dim=config.input_dim, action_dim=config.action_dim, hidden_dims=config.hidden_dims)
        self._head = MLPActionHead(head_config)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if features is None and pixel_values is not None:
            x = self._pool(pixel_values).flatten(1)
            if x.shape[1] != self.config.input_dim:
                x = nn.functional.adaptive_avg_pool1d(x.unsqueeze(1), self.config.input_dim).squeeze(1)
            features = x
        action = self._head(features)
        return {"action": action, "logits": action}
