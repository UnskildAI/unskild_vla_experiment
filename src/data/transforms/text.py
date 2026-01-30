"""Text transforms / tokenization (dataset-agnostic interface)."""

from __future__ import annotations

from typing import Any, Callable

import torch

from src.core.typing import TensorDict


def build_text_transforms(
    max_length: int = 512,
    padding: str = "max_length",
    truncation: bool = True,
    tokenizer: Any = None,
    extra: dict[str, Any] | None = None,
) -> Callable[[str | list[int]], TensorDict]:
    """Build text transform (tokenizer wrapper). Returns callable text -> dict of tensors."""
    extra = extra or {}

    def _apply(text: str | list[int]) -> TensorDict:
        if tokenizer is not None:
            out = tokenizer(
                text,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors="pt",
            )
            return {
                "input_ids": out["input_ids"].squeeze(0),
                "attention_mask": out["attention_mask"].squeeze(0),
            }
        # No tokenizer: treat as already tokenized or placeholder
        if isinstance(text, str):
            ids = [0] * max_length
        else:
            ids = list(text)[:max_length]
            ids = ids + [0] * (max_length - len(ids))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(max_length, dtype=torch.long),
        }

    return _apply
