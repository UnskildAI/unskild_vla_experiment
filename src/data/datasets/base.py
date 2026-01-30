"""BaseDataset: vision, text, action, multimodal — ABC + torch-native, Pydantic config."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset
from pydantic import BaseModel

from src.core.typing import BatchDict


class BaseDatasetConfig(BaseModel):
    """Base config for datasets."""

    extra: dict[str, Any] = {}


class BaseDataset(TorchDataset[BatchDict], ABC):
    """Abstract base dataset: vision, text, action, or multimodal. Torch-native, config-driven."""

    def __init__(self, config: BaseDatasetConfig, **kwargs: Any) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> BatchDict:
        """Return a batch-ready dict (e.g. images, texts, actions, masks)."""
        pass

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Optional: dataset version / hash for reproducibility."""
        return {"dataset_config": self.config.model_dump()}
