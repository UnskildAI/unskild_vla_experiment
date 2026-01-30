"""State-action / trajectory datasets for vision → action policy learning."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, Field

from src.data.datasets.base import BaseDataset, BaseDatasetConfig
from src.core.typing import BatchDict


class ActionDatasetConfig(BaseDatasetConfig):
    """Config for state-action / trajectory datasets."""

    state_dim: int = Field(0, ge=0)
    action_dim: int = Field(0, gt=0)
    sequence_length: int = Field(1, gt=0)
    extra: dict[str, Any] = Field(default_factory=dict)


class ActionDataset(BaseDataset):
    """State-action or trajectory dataset. Returns vision (or state) + action tensors."""

    def __init__(self, config: ActionDatasetConfig | dict[str, Any], **kwargs: Any) -> None:
        if isinstance(config, dict):
            config = ActionDatasetConfig(**config)
        super().__init__(config)
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.sequence_length = config.sequence_length
        self._items: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> BatchDict:
        item = self._items[index] if index < len(self._items) else {}
        # Placeholder: image/state and action
        image = item.get("image", torch.zeros(3, 224, 224))
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        action = item.get("action", torch.zeros(self.action_dim))
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        return {
            "pixel_values": image,
            "action": action,
            "action_mask": item.get("action_mask", torch.ones_like(action)),
        }
