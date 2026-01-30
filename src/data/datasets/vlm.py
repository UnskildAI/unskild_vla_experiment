"""Image-text / OCR / caption datasets for VLM finetuning."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, Field

from src.data.datasets.base import BaseDataset, BaseDatasetConfig
from src.core.typing import BatchDict


class VLMDatasetConfig(BaseDatasetConfig):
    """Config for VLM (image-text) datasets."""

    max_length: int = Field(512, gt=0)
    image_size: int = Field(224, gt=0)
    extra: dict[str, Any] = Field(default_factory=dict)


class VLMDataset(BaseDataset):
    """Image-text / caption dataset. Returns vision + text tensors for VLM."""

    def __init__(self, config: VLMDatasetConfig | dict[str, Any], **kwargs: Any) -> None:
        if isinstance(config, dict):
            config = VLMDatasetConfig(**config)
        super().__init__(config)
        self.max_length = config.max_length
        self.image_size = config.image_size
        # Placeholder: in practice load from disk/API
        self._items: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> BatchDict:
        item = self._items[index] if index < len(self._items) else {}
        # Placeholder tensors; real impl would load image + tokenize text
        image = item.get("image", torch.zeros(3, self.image_size, self.image_size))
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        input_ids = item.get("input_ids", torch.zeros(self.max_length, dtype=torch.long))
        attention_mask = item.get("attention_mask", torch.ones(self.max_length, dtype=torch.long))
        labels = item.get("labels", input_ids.clone())
        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
