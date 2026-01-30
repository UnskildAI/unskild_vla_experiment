"""Action metrics (e.g. L2 error, success rate placeholders)."""

from __future__ import annotations

from typing import Any

import torch

from src.core.typing import TensorDict


def action_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    **kwargs: Any,
) -> dict[str, float]:
    """Compute action metrics: L2 error, etc."""
    with torch.no_grad():
        l2 = ((pred - target) ** 2).mean().sqrt().item()
        mae = (pred - target).abs().mean().item()
    return {"l2_error": l2, "mae": mae}
