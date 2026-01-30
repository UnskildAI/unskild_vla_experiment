"""Vision metrics (e.g. FID, LPIPS placeholders)."""

from __future__ import annotations

from typing import Any

import torch

from src.core.typing import TensorDict


def vision_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    **kwargs: Any,
) -> dict[str, float]:
    """Compute vision metrics. Placeholder: MSE; FID/LPIPS when available."""
    with torch.no_grad():
        mse = ((pred - target) ** 2).mean().item()
    return {"mse": mse}
