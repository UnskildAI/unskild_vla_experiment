"""Shared Protocols and type aliases."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

import torch

# Type aliases for common structures
TensorDict = dict[str, torch.Tensor]
BatchDict = dict[str, Any]  # vision, text, action, etc.

ConfigT = TypeVar("ConfigT", bound=Any)


class Configurable(Protocol[ConfigT]):
    """Protocol for components that accept a Pydantic config."""

    def __init__(self, config: ConfigT, **kwargs: Any) -> None: ...
    def get_config(self) -> ConfigT: ...


class Checkpointable(Protocol):
    """Protocol for components that support save/load checkpoint."""

    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None: ...
