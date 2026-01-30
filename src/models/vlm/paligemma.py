"""PaliGemma-style VLM wrapper (HuggingFace transformers when available)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from src.models.vlm.base import BaseVLM, BaseVLMConfig
from src.core.typing import TensorDict


from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, PaliGemmaConfig as HFPaliGemmaConfig

class PaliGemmaConfig(BaseVLMConfig):
    """Config for PaliGemma-style VLM."""

    model_name_or_path: str = Field("google/paligemma-3b-pt-224", description="HF model id or path")
    vision_frozen: bool = False
    language_frozen: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class PaliGemmaVLM(BaseVLM):
    """PaliGemma VLM using Hugging Face transformers."""

    def __init__(self, config: PaliGemmaConfig | dict[str, Any], **kwargs: Any) -> None:
        if isinstance(config, dict):
            config = PaliGemmaConfig(**config)
        super().__init__(config)
        self.model_name_or_path = config.model_name_or_path
        
        # Load real model from Hugging Face
        # Note: This requires authentication for gated models (paligemma isn't gated usually, but check)
        # Using float32 for CPU compatibility as default, but ideally float16/bfloat16 for GPU
        self.hf_model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float32, 
            low_cpu_mem_usage=True
        )
        self.processor = PaliGemmaProcessor.from_pretrained(self.model_name_or_path)

        if config.vision_frozen:
            self.freeze_vision()
        if config.language_frozen:
            self.freeze_language()

    def freeze_vision(self) -> None:
        for param in self.hf_model.vision_tower.parameters():
            param.requires_grad = False
        for param in self.hf_model.multi_modal_projector.parameters():
            param.requires_grad = False

    def freeze_language(self) -> None:
        for param in self.hf_model.language_model.parameters():
            param.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Expects pixel_values: (B, C, H, W)
        # PaliGemma expects normalized pixels in roughly [-1, 1] range usually handle by processor
        # Here assuming data loader gives correct preprocessed tensors
        # vision_tower outputs (B, Seq, Hidden)
        return self.hf_model.vision_tower(pixel_values).last_hidden_state

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Using the language model's embedding layer directly isn't enough for PaliGemma as it's a full decoder
        # But for 'encode_text' abstract method, we can return embeddings
        # However, PaliGemma's forward pass handles fusion internally.
        # Use embeddings for compatibility if needed, but preferred to use forward() directly
        return self.hf_model.language_model.model.embed_tokens(input_ids)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> TensorDict:
        # Handles the full forward pass
        # pixel_values: (B, C, H, W)
        # input_ids: (B, Seq)
        # attention_mask: (B, Seq)
        # labels: (B, Seq)
        
        if input_ids is not None:
             input_ids = input_ids.long()
        if labels is not None:
             labels = labels.long()
        if attention_mask is not None:
             attention_mask = attention_mask.long()

        # The HF model handles multimodal fusion
        outputs = self.hf_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            # embeddings are not directly returned by default forward, but we can extract if really needed
            # for now, relying on logits/loss from the model itself
        }
