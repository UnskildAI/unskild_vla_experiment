"""Vision encoders: registry and build function."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.core.registry import _Registry

VisionEncoderRegistry: _Registry = _Registry()


def build_vision_encoder(name: str, config: Any, **kwargs: Any) -> nn.Module:
    """Build vision encoder from registry."""
    return VisionEncoderRegistry.build(name, config, **kwargs)
