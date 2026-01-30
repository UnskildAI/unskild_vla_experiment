"""Run evaluation benchmarks."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from src.core.logging import get_logger
from src.core.typing import BatchDict


def run_benchmark(model: torch.nn.Module, dataloader: DataLoader, metric_fns: dict[str, Any], device: torch.device | None = None, **kwargs: Any) -> dict[str, float]:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    model.eval()
    agg: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(**{k: v for k, v in batch.items() if k in ("pixel_values", "input_ids", "attention_mask", "image")})
            for name, fn in metric_fns.items():
                m = fn(out, batch, **kwargs)
                for k, v in m.items():
                    key = f"{name}/{k}" if name else k
                    agg.setdefault(key, []).append(v)
    result = {k: sum(v) / len(v) for k, v in agg.items()}
    get_logger("benchmark").info("Benchmark result: %s", result)
    return result
