"""Image/video datasets for flow-matching and diffusion training."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, Field

from src.data.datasets.base import BaseDataset, BaseDatasetConfig
from src.core.typing import BatchDict


class DiffusionDatasetConfig(BaseDatasetConfig):
    """Config for diffusion/flow-matching image or video datasets."""

    image_size: int = Field(64, gt=0)
    num_channels: int = Field(3, gt=0)
    conditioning: str = Field("none", description="none, image, text, action")
    extra: dict[str, Any] = Field(default_factory=dict)


class DiffusionDataset(BaseDataset):
    """Image (or video) dataset for flow matching / diffusion. Optional conditioning."""

    def __init__(self, config: DiffusionDatasetConfig | dict[str, Any], **kwargs: Any) -> None:
        if isinstance(config, dict):
            config = DiffusionDatasetConfig(**config)
        super().__init__(config)
        self.image_size = config.image_size
        self.num_channels = config.num_channels
        self.conditioning = config.conditioning
        self._items: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> BatchDict:
        item = self._items[index] if index < len(self._items) else {}
        # Placeholder: clean image; conditioning optional
        image = item.get("image", torch.rand(self.num_channels, self.image_size, self.image_size))
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        out: BatchDict = {"image": image}
        if self.conditioning == "text":
            out["conditioning_text"] = item.get("text", "")
        if self.conditioning == "image":
            out["conditioning_image"] = item.get("cond_image", image)
        if self.conditioning == "action":
            out["conditioning_action"] = item.get("action", torch.zeros(1))
        return out
