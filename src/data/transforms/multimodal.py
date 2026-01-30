"""Multimodal transforms: compose vision + text for VLM batches."""

from __future__ import annotations

from typing import Any, Callable

from src.core.typing import BatchDict, TensorDict


def build_multimodal_transforms(
    vision_transform: Callable[[Any], TensorDict] | None = None,
    text_transform: Callable[[Any], TensorDict] | None = None,
    extra: dict[str, Any] | None = None,
) -> Callable[[Any, Any], BatchDict]:
    """Build a transform that applies vision and text transforms and merges into one batch dict."""
    extra = extra or {}
    vision_transform = vision_transform or (lambda x: {})
    text_transform = text_transform or (lambda x: {})

    def _apply(image: Any, text: Any) -> BatchDict:
        out: BatchDict = {}
        out.update(vision_transform(image))
        out.update(text_transform(text))
        return out

    return _apply
