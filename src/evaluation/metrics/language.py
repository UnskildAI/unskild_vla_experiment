"""Language metrics (e.g. perplexity, BLEU placeholders)."""

from __future__ import annotations

from typing import Any

import torch

from src.core.typing import TensorDict


def language_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    **kwargs: Any,
) -> dict[str, float]:
    """Compute language metrics. Placeholder: cross-entropy / perplexity."""
    with torch.no_grad():
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index,
            reduction="mean",
        ).item()
        ppl = float(torch.exp(torch.tensor(ce)).item())
    return {"cross_entropy": ce, "perplexity": ppl}
