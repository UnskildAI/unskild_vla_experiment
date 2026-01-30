"""BaseVLM: encode_image, encode_text, forward — ABC + torch-native, Pydantic config, checkpoint."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel

from src.core.typing import BatchDict, TensorDict


class BaseVLMConfig(BaseModel):
    """Base config for VLMs."""

    vision_frozen: bool = False
    language_frozen: bool = False
    extra: dict[str, Any] = {}


class BaseVLM(nn.Module, ABC):
    """Abstract base VLM: encode_image, encode_text, forward. Torch-native, config-driven, checkpointable."""

    def __init__(self, config: BaseVLMConfig, **kwargs: Any) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to vision embeddings."""
        pass

    @abstractmethod
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode text to language embeddings."""
        pass

    @abstractmethod
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> TensorDict:
        """Full forward: vision + language -> logits / embeddings."""
        pass

    def freeze_vision(self) -> None:
        """Freeze vision encoder parameters."""
        for p in self.parameters():
            if "vision" in (getattr(m, "name", "") for m in [p] if hasattr(p, "name")):
                p.requires_grad = False
        # Fallback: submodules named *vision* or *encoder* (image)
        for name, child in self.named_children():
            if "vision" in name.lower() or "encoder" in name.lower():
                for p in child.parameters():
                    p.requires_grad = False

    def freeze_language(self) -> None:
        """Freeze language decoder parameters."""
        for name, child in self.named_children():
            if "language" in name.lower() or "text" in name.lower() or "decoder" in name.lower():
                for p in child.parameters():
                    p.requires_grad = False

    def get_config(self) -> BaseVLMConfig:
        return self.config
