"""BaseTrainer: train, eval, save, load — ABC + explicit training loop, gradient accumulation, mixed precision."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from pydantic import BaseModel

from src.core.typing import BatchDict
from src.core.exceptions import VLATrainingError, VLACheckpointError
from src.core.logging import get_logger


class BaseTrainerConfig(BaseModel):
    """Base config for trainers."""

    max_steps: int = 10_000
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    max_grad_norm: float | None = None
    log_interval: int = 10
    eval_interval: int = 500
    checkpoint_interval: int = 1000
    output_dir: str = "./outputs"
    seed: int = 42
    extra: dict[str, Any] = {}


class BaseTrainer(ABC):
    """Abstract base trainer: explicit training loop, gradient accumulation, mixed precision, checkpoint save/load."""

    def __init__(self, config: BaseTrainerConfig, **kwargs: Any) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.global_step = 0
        self._scaler: torch.amp.GradScaler | None = None

    @abstractmethod
    def train_step(self, batch: BatchDict) -> dict[str, Any]:
        """Single training step. Must include 'loss' (tensor) and may include scalar metrics for logging."""
        pass

    @abstractmethod
    def eval_step(self, batch: BatchDict) -> dict[str, float]:
        """Single eval step. Returns dict of scalar metrics."""
        pass

    def train(
        self,
        train_loader: DataLoader[BatchDict],
        eval_loader: DataLoader[BatchDict] | None = None,
        callbacks: list[Any] | None = None,
    ) -> None:
        """Explicit training loop: gradient accumulation, mixed precision, no magic helpers."""
        callbacks = callbacks or []
        self._setup_mixed_precision()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model.to(self.device)
        
        accum_steps = self.config.gradient_accumulation_steps
        max_grad_norm = getattr(self.config, "max_grad_norm", None)
        self.model.train()
        iterator = iter(train_loader)
        for step in range(self.config.max_steps):
            loss_accum: dict[str, Any] = {}
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(accum_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                    
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                with torch.amp.autocast("cuda", enabled=self.config.mixed_precision != "no"):
                    step_metrics = self.train_step(batch)
                loss = step_metrics.get("loss")
                if loss is not None and isinstance(loss, torch.Tensor):
                    loss = loss / accum_steps
                    if self._scaler:
                        self._scaler.scale(loss).backward()
                    else:
                        loss.backward()
                for k, v in step_metrics.items():
                    if k == "loss" and isinstance(v, torch.Tensor):
                        v = v.detach().item()
                    if isinstance(v, (int, float)):
                        loss_accum[k] = loss_accum.get(k, 0.0) + v
            for k in loss_accum:
                loss_accum[k] /= accum_steps
            if max_grad_norm is not None:
                if self._scaler:
                    self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            if self._scaler:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()
            self.global_step += 1
            if self.global_step % self.config.log_interval == 0:
                self.logger.info("step=%d %s", self.global_step, loss_accum)
            for cb in callbacks:
                cb.on_step_end(self, step=self.global_step, metrics=loss_accum)
            if eval_loader and self.global_step % self.config.eval_interval == 0:
                self.evaluate(eval_loader)
            if self.global_step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(Path(self.config.output_dir) / f"ckpt_{self.global_step}.pt")
        self.logger.info("Training finished. total_steps=%d", self.global_step)

    def _setup_mixed_precision(self) -> None:
        if self.config.mixed_precision == "fp16":
            self._scaler = torch.amp.GradScaler("cuda")
        else:
            self._scaler = None

    def evaluate(self, eval_loader: DataLoader[BatchDict]) -> dict[str, float]:
        """Run evaluation over eval_loader."""
        self.model.eval()
        agg: dict[str, list[float]] = {}
        with torch.no_grad():
            for batch in eval_loader:
                m = self.eval_step(batch)
                for k, v in m.items():
                    agg.setdefault(k, []).append(v)
        self.model.train()
        return {k: sum(v) / len(v) for k, v in agg.items()}

    def save_checkpoint(self, path: Path | str) -> None:
        """Save trainer/model/optimizer state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self._scaler is not None:
            state["scaler_state_dict"] = self._scaler.state_dict()
        torch.save(state, path)
        self.logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: Path | str) -> None:
        """Load trainer/model/optimizer state."""
        path = Path(path)
        if not path.exists():
            raise VLACheckpointError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        if "optimizer_state_dict" in state and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_step = state.get("global_step", 0)
        if self._scaler is not None and "scaler_state_dict" in state:
            self._scaler.load_state_dict(state["scaler_state_dict"])
        self.logger.info("Checkpoint loaded: %s", path)

    @property
    def model(self) -> torch.nn.Module:
        """Subclasses must set _model."""
        if not hasattr(self, "_model"):
            raise VLATrainingError("Trainer has no model. Set self._model in subclass.")
        return self._model

    @model.setter
    def model(self, value: torch.nn.Module) -> None:
        self._model = value

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if not hasattr(self, "_optimizer"):
            raise VLATrainingError("Trainer has no optimizer. Set self._optimizer in subclass.")
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: torch.optim.Optimizer) -> None:
        self._optimizer = value
