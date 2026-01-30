"""LLaVA-style VLM wrapper (HuggingFace when available)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from src.models.vlm.base import BaseVLM, BaseVLMConfig
from src.core.typing import TensorDict


class LLaVAConfig(BaseVLMConfig):
    """Config for LLaVA-style VLM."""

    model_name_or_path: str = Field("llava-hf/llava-1.5-7b-hf", description="HF model id or path")
    vision_frozen: bool = False
    language_frozen: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class LLaVAVLM(BaseVLM):
    """LLaVA-style VLM. Stub for scaffolding; real impl loads from transformers."""

    def __init__(self, config: LLaVAConfig | dict[str, Any], **kwargs: Any) -> None:
        if isinstance(config, dict):
            config = LLaVAConfig(**config)
        super().__init__(config)
        self.model_name_or_path = config.model_name_or_path
        self._vision_proj = nn.Linear(1024, 4096)
        self._language_proj = nn.Linear(4096, 32000)
        self._hidden_size = 4096
        if config.vision_frozen:
            self.freeze_vision()
        if config.language_frozen:
            self.freeze_language()

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pixel_values.shape
        x = pixel_values.flatten(1)
        if x.shape[1] != 1024:
            x = nn.functional.adaptive_avg_pool1d(x.unsqueeze(1), 1024).squeeze(1)
        return self._vision_proj(x)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        emb = torch.nn.functional.one_hot(input_ids.clamp(0, 31999), 32000).float()
        emb = emb.matmul(torch.eye(32000, self._hidden_size, device=input_ids.device, dtype=input_ids.dtype))
        return emb.mean(dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> TensorDict:
        vision_emb = self.encode_image(pixel_values) if pixel_values is not None else None
        text_emb = self.encode_text(input_ids, attention_mask) if input_ids is not None else None
        inp = vision_emb if vision_emb is not None else text_emb
        logits = self._language_proj(inp)
        return {"logits": logits, "vision_embeddings": vision_emb, "language_embeddings": text_emb}
