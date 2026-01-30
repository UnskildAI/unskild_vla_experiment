"""PaliGemma-style VLM wrapper (HuggingFace transformers when available)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from src.models.vlm.base import BaseVLM, BaseVLMConfig
from src.core.typing import TensorDict


class PaliGemmaConfig(BaseVLMConfig):
    """Config for PaliGemma-style VLM."""

    model_name_or_path: str = Field("google/paligemma-3b-pt-224", description="HF model id or path")
    vision_frozen: bool = False
    language_frozen: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class PaliGemmaVLM(BaseVLM):
    """PaliGemma-style VLM. Uses HF AutoModel when available; else stub for scaffolding."""

    def __init__(self, config: PaliGemmaConfig | dict[str, Any], **kwargs: Any) -> None:
        if isinstance(config, dict):
            config = PaliGemmaConfig(**config)
        super().__init__(config)
        self.model_name_or_path = config.model_name_or_path
        # Stub: real impl would load from transformers
        self._vision_proj = nn.Linear(768, 2048)
        self._language_proj = nn.Linear(2048, 32000)
        self._hidden_size = 2048
        if config.vision_frozen:
            self.freeze_vision()
        if config.language_frozen:
            self.freeze_language()

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pixel_values.shape
        # Placeholder: flatten and project
        x = pixel_values.flatten(1)
        if x.shape[1] != 768:
            x = nn.functional.adaptive_avg_pool1d(x.unsqueeze(1), 768).squeeze(1)
        return self._vision_proj(x)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Placeholder: embed then pool
        emb = torch.nn.functional.one_hot(input_ids.clamp(0, 31999), 32000).float().matmul(
            torch.eye(32000, 2048, device=input_ids.device, dtype=torch.float32)
        )
        return emb

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> TensorDict:
        if pixel_values is not None:
            vision_emb = self.encode_image(pixel_values)
        else:
            vision_emb = None
        if input_ids is not None:
            text_emb = self.encode_text(input_ids, attention_mask)
        else:
            text_emb = None
        logits = self._language_proj(text_emb if text_emb is not None else vision_emb)
        return {"logits": logits, "vision_embeddings": vision_emb, "language_embeddings": text_emb}
