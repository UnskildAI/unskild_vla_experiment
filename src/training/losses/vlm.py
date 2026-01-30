"""VLM loss: language modeling / cross-entropy."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from src.core.typing import TensorDict


class VLMLossConfig(BaseModel):
    """Config for VLM loss."""

    label_smoothing: float = Field(0.0, ge=0, le=1)
    ignore_index: int = -100
    extra: dict[str, Any] = Field(default_factory=dict)


class VLMLoss(nn.Module):
    """Language modeling loss (cross-entropy over logits vs labels)."""

    def __init__(self, config: VLMLossConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.config = config or VLMLossConfig(**kwargs)
        self._ce = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing,
            ignore_index=self.config.ignore_index,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # logits: (B, seq, vocab), labels: (B, seq)
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
        return self._ce(logits, labels)
