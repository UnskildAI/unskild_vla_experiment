"""Flow-matching / diffusion trainer: explicit loop, continuous time, pluggable loss."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from pydantic import BaseModel, Field

from src.training.trainers.base import BaseTrainer, BaseTrainerConfig
from src.training.losses.diffusion import FlowMatchingLoss, FlowMatchingLossConfig
from src.training.optimizers import build_optimizer, OptimizerConfig
from src.core.typing import BatchDict
from src.models.diffusion.schedulers import FlowMatchScheduler


class DiffusionTrainerConfig(BaseTrainerConfig):
    """Config for diffusion trainer."""

    learning_rate: float = Field(1e-4, gt=0)
    weight_decay: float = Field(0.0, ge=0)
    max_grad_norm: float | None = Field(None)
    extra: dict[str, Any] = Field(default_factory=dict)


class DiffusionTrainer(BaseTrainer):
    """Flow-matching / diffusion training: explicit loop, continuous time, pluggable loss."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: DiffusionTrainerConfig,
        loss_config: FlowMatchingLossConfig | None = None,
        scheduler: FlowMatchScheduler | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        self._model = model
        self._loss_fn = FlowMatchingLoss(loss_config or FlowMatchingLossConfig())
        self._flow_scheduler = scheduler or FlowMatchScheduler()
        opt_config = OptimizerConfig(name="adamw", lr=config.learning_rate, weight_decay=config.weight_decay)
        self._optimizer = build_optimizer(model.parameters(), opt_config)

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
        image = batch.get("image")
        if image is None:
            raise KeyError("Diffusion batch must contain 'image'")
        device = image.device
        b = image.shape[0]
        t = self._flow_scheduler.sample_t(b, device)
        # Flow matching: x0 = image, x1 = noise (or vice versa); xt = (1-t)*x0 + t*x1, target = velocity
        x1 = torch.randn_like(image, device=device)
        xt = self._flow_scheduler.interpolate(image, x1, t)
        target = self._flow_scheduler.velocity(image, x1)
        conditioning = {k: v for k, v in batch.items() if k not in ("image",) and isinstance(v, torch.Tensor)}
        pred = self._model(xt, t, conditioning=conditioning or None)
        loss = self._loss_fn(pred=pred, target=target)
        return {"loss": loss, "loss_value": loss.detach().item()}

    def eval_step(self, batch: BatchDict) -> dict[str, float]:
        with torch.no_grad():
            out = self.train_step(batch)
        return {"loss": out["loss_value"]}
