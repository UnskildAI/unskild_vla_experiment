"""Noise / flow schedulers for diffusion and flow-matching (continuous time)."""

from __future__ import annotations

from typing import Any

import torch


class FlowMatchScheduler:
    """Flow-matching scheduler: interpolate x0 -> x1 with t in [0,1]. Explicit noise schedule optional."""

    def __init__(self, sigma_min: float = 0.0, sigma_max: float = 1.0) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample time t ~ U(0,1)."""
        return torch.rand(batch_size, device=device)

    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """xt = (1-t)*x0 + t*x1. t shape (B,) broadcast to x."""
        while t.dim() < x0.dim():
            t = t.view(*t.shape, *([1] * (x0.dim() - t.dim())))
        return (1 - t) * x0 + t * x1

    def velocity(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Constant velocity dx/dt = x1 - x0."""
        return x1 - x0
