"""VLM finetuning trainer."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, Field

from src.training.trainers.base import BaseTrainer, BaseTrainerConfig
from src.training.losses.vlm import VLMLoss, VLMLossConfig
from src.training.optimizers import build_optimizer, OptimizerConfig
from src.core.typing import BatchDict

from transformers.optimization import Adafactor


class VLMTrainerConfig(BaseTrainerConfig):
    learning_rate: float = Field(1e-5, gt=0)
    weight_decay: float = Field(0.0, ge=0)
    max_grad_norm: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

class VLMTrainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, config: VLMTrainerConfig, loss_config: VLMLossConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config)
        self._model = model
        self._loss_fn = VLMLoss(loss_config or VLMLossConfig())
        
        # Use Adafactor for memory efficiency (critical for 3B+ models on 24GB GPUs)
        # Disable scaling/relative_step for fine-tuning with explicit LR
        self._optimizer = Adafactor(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            scale_parameter=False,
            relative_step=False,
        )

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @model.setter
    def model(self, value: torch.nn.Module) -> None:
        self._model = value

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: torch.optim.Optimizer) -> None:
        self._optimizer = value

    def train_step(self, batch: BatchDict) -> dict[str, Any]:
        out = self._model(pixel_values=batch.get("pixel_values"), input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask"))
        logits = out.get("logits")
        labels = batch.get("labels")
        if labels is None:
            labels = batch.get("input_ids")
        loss = self._loss_fn(logits=logits, labels=labels, attention_mask=batch.get("attention_mask"))
        return {"loss": loss, "loss_value": loss.detach().item()}

    def eval_step(self, batch: BatchDict) -> dict[str, float]:
        with torch.no_grad():
            out = self.train_step(batch)
        return {"loss": out["loss_value"]}
