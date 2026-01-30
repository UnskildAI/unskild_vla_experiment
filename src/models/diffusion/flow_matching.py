"""Flow-matching model: continuous time, pluggable loss, conditioning."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from src.models.diffusion.base import BaseDiffusionModel, BaseDiffusionConfig
from src.core.typing import TensorDict


class FlowMatchingConfig(BaseDiffusionConfig):
    """Config for flow-matching model."""

    in_channels: int = Field(3, gt=0)
    out_channels: int = Field(3, gt=0)
    hidden_dim: int = Field(256, gt=0)
    conditioning_dim: int = Field(0, ge=0)
    num_blocks: int = Field(4, gt=0)
    extra: dict[str, Any] = Field(default_factory=dict)


class FlowMatchingModel(BaseDiffusionModel):
    """Flow-matching model: predicts velocity field. Continuous time, optional conditioning."""

    def __init__(self, config: FlowMatchingConfig | dict[str, Any], **kwargs: Any) -> None:
        if isinstance(config, dict):
            config = FlowMatchingConfig(**config)
        super().__init__(config)
        self.hidden_dim = config.hidden_dim
        self.num_blocks = config.num_blocks
        cin = config.in_channels + 1  # +1 for time
        if config.conditioning_dim > 0:
            cin += config.conditioning_dim
        self._net = nn.Sequential(
            nn.Conv2d(cin, config.hidden_dim, 3, padding=1),
            nn.SiLU(),
            *[
                nn.Sequential(
                    nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
                    nn.SiLU(),
                )
                for _ in range(config.num_blocks - 1)
            ],
            nn.Conv2d(config.hidden_dim, config.out_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: TensorDict | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # t: (B,) -> (B,1,1,1) for broadcast
        while t.dim() < 4:
            t = t.unsqueeze(-1)
        inp = torch.cat([x, t.expand(-1, 1, x.shape[2], x.shape[3])], dim=1)
        if conditioning and self.config.conditioning_dim > 0:
            c = conditioning.get("embedding")
            if c is not None:
                c = c.view(c.shape[0], -1, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
                inp = torch.cat([inp, c], dim=1)
        return self._net(inp)
