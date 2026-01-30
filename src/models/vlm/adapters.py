"""LoRA, QLoRA, prefix tuning adapters for VLMs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from src.models.vlm.base import BaseVLM, BaseVLMConfig


class LoRAConfig(BaseModel):
    """LoRA adapter config."""

    r: int = Field(8, gt=0, description="Rank")
    alpha: float = Field(16.0, gt=0)
    dropout: float = Field(0.0, ge=0, le=1)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    extra: dict[str, Any] = Field(default_factory=dict)


class LoRAAdapter(nn.Module):
    """LoRA adapter: low-rank projection for linear layers. Wrapped around base VLM."""

    def __init__(self, base: BaseVLM, config: LoRAConfig, **kwargs: Any) -> None:
        super().__init__()
        self.base = base
        self.config = config
        self._lora_layers: nn.ModuleDict = nn.ModuleDict()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.base(*args, **kwargs)


class QLoRAConfig(BaseModel):
    """QLoRA adapter config (quantized base + LoRA)."""

    bits: int = Field(4, description="Quantization bits")
    r: int = Field(8, gt=0)
    alpha: float = Field(16.0, gt=0)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    extra: dict[str, Any] = Field(default_factory=dict)


class QLoRAAdapter(nn.Module):
    """QLoRA: quantized base model + LoRA. Stub; real impl uses bitsandbytes etc."""

    def __init__(self, base: BaseVLM, config: QLoRAConfig, **kwargs: Any) -> None:
        super().__init__()
        self.base = base
        self.config = config

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.base(*args, **kwargs)
