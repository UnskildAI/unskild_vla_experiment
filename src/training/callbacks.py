"""Callbacks: logging, EMA, checkpointing. No magic training hidden in helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.core.registry import _Registry

CallbackRegistry: _Registry = _Registry()


class BaseCallback(ABC):
    """Base callback: on_step_end, on_epoch_end, etc. Explicit hooks only."""

    @abstractmethod
    def on_step_end(self, trainer: Any, step: int, metrics: dict[str, Any]) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, Any]) -> None:
        pass

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass


class CheckpointCallback(BaseCallback):
    """Save checkpoint every N steps. Explicit, no magic."""

    def __init__(self, save_dir: str | Path, interval: int = 1000, keep_last_n: int = 3) -> None:
        self.save_dir = Path(save_dir)
        self.interval = interval
        self.keep_last_n = keep_last_n
        self._saved: list[Path] = []

    def on_step_end(self, trainer: Any, step: int, metrics: dict[str, Any]) -> None:
        if step > 0 and step % self.interval == 0:
            path = self.save_dir / f"ckpt_{step}.pt"
            trainer.save_checkpoint(path)
            self._saved.append(path)
            while len(self._saved) > self.keep_last_n:
                old = self._saved.pop(0)
                if old.exists():
                    old.unlink()


class LoggingCallback(BaseCallback):
    """Log metrics to stdout (or W&B / TensorBoard if configured). Explicit hooks."""

    def __init__(self, log_interval: int = 10) -> None:
        self.log_interval = log_interval

    def on_step_end(self, trainer: Any, step: int, metrics: dict[str, Any]) -> None:
        if step > 0 and step % self.log_interval == 0:
            trainer.logger.info("step=%d %s", step, metrics)
