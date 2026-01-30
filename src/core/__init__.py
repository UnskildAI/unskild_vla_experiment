"""Core utilities: registry, typing, exceptions, logging."""

from src.core.registry import (
    DatasetRegistry,
    ModelRegistry,
    LossRegistry,
    TrainerRegistry,
)
from src.core.typing import (
    BatchDict,
    TensorDict,
    ConfigT,
)
from src.core.exceptions import (
    VLAConfigError,
    VLACheckpointError,
    VLADataError,
    VLATrainingError,
)
from src.core.logging import get_logger, setup_logging

__all__ = [
    "DatasetRegistry",
    "ModelRegistry",
    "LossRegistry",
    "TrainerRegistry",
    "BatchDict",
    "TensorDict",
    "ConfigT",
    "VLAConfigError",
    "VLACheckpointError",
    "VLADataError",
    "VLATrainingError",
    "get_logger",
    "setup_logging",
]
