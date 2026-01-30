"""Pydantic-based configuration schema with YAML loading and env expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from src.core.exceptions import VLAConfigError


def _expand_env(value: str) -> str:
    """Expand ${VAR} and $VAR in string using environment."""
    return re.sub(
        r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
        lambda m: os.environ.get(m.group(1) or m.group(2) or "", ""),
        value,
    )


def _expand_dict(obj: Any) -> Any:
    """Recursively expand env vars in dict/list/str."""
    if isinstance(obj, dict):
        return {k: _expand_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_dict(x) for x in obj]
    if isinstance(obj, str):
        return _expand_env(obj)
    return obj


class ModelConfig(BaseModel):
    """Model component configuration (VLM, diffusion, action)."""

    name: str = Field("vlm", description="Model registry name")
    frozen: bool = Field(False, description="Whether to freeze this component")
    extra: dict[str, Any] = Field(default_factory=dict)


class DataConfig(BaseModel):
    """Dataset and dataloader configuration."""

    dataset_name: str = Field("vlm", description="Dataset registry name")
    batch_size: int = Field(32, gt=0)
    num_workers: int = Field(0, ge=0)
    shuffle: bool = True
    drop_last: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Training loop configuration."""

    max_steps: int = Field(10_000, gt=0)
    gradient_accumulation_steps: int = Field(1, gt=0)
    mixed_precision: str = Field("fp16", description="fp16, bf16, or no")
    seed: int = 42
    log_interval: int = 10
    eval_interval: int = 500
    checkpoint_interval: int = 1000
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float | None = Field(None, description="Gradient clipping")
    extra: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Top-level experiment config: model, data, training, reproducibility."""

    experiment_name: str = "default"
    output_dir: str = "./outputs"
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    seed: int = 42
    model_hash: str | None = None
    dataset_version: str | None = None


def load_config(path: str | Path, expand_env: bool = True) -> ExperimentConfig:
    """Load YAML config from path and optionally expand env vars."""
    path = Path(path)
    if not path.exists():
        raise VLAConfigError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if expand_env:
        raw = _expand_dict(raw)
    return ExperimentConfig.model_validate(raw)
