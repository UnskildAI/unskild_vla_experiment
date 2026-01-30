"""BasePipeline: compose stages. ABC + config, checkpoint."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class BasePipelineConfig(BaseModel):
    experiment_name: str = "default"
    output_dir: str = "./outputs"
    seed: int = 42
    extra: dict[str, Any] = {}


class BasePipeline(ABC):
    def __init__(self, config: BasePipelineConfig, **kwargs: Any) -> None:
        self.config = config

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        pass

    def save_checkpoint(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self, path: Path | str) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
