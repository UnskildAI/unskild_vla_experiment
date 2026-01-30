"""Vision transforms for images (resize, normalize, augment)."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torchvision import transforms as TV

from src.core.typing import TensorDict


def build_vision_transforms(
    image_size: int = 224,
    is_training: bool = True,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
    extra: dict[str, Any] | None = None,
) -> Callable[[Any], TensorDict]:
    """Build a vision transform pipeline. Returns callable that maps raw image -> dict of tensors."""
    extra = extra or {}
    pipeline: list[Any] = [
        TV.Resize((image_size, image_size)),
        TV.ToTensor(),
        TV.Normalize(mean=mean, std=std),
    ]
    if is_training and extra.get("augment", False):
        pipeline.insert(1, TV.RandomHorizontalFlip())
    transform = TV.Compose(pipeline)

    def _apply(img: Any) -> TensorDict:
        if not isinstance(img, torch.Tensor):
            img = TV.functional.to_tensor(img)
        return {"pixel_values": transform(img)}

    return _apply
