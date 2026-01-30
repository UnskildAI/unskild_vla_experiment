"""Language decoders: registry and build function."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.core.registry import _Registry

LanguageDecoderRegistry: _Registry = _Registry()


def build_language_decoder(name: str, config: Any, **kwargs: Any) -> nn.Module:
    """Build language decoder from registry."""
    return LanguageDecoderRegistry.build(name, config, **kwargs)
