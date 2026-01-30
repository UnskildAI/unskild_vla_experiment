"""Build DataLoader from config and dataset."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from src.core.typing import BatchDict
from src.core.registry import DatasetRegistry
from src.core.exceptions import VLADataError


def build_dataloader(
    dataset_name: str,
    dataset_config: Any,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = False,
    **kwargs: Any,
) -> DataLoader[BatchDict]:
    """Build a DataLoader from registry dataset name and config."""
    if dataset_name not in DatasetRegistry:
        raise VLADataError(f"Unknown dataset: {dataset_name}. Registered: {list(DatasetRegistry.keys())}")
    dataset = DatasetRegistry.build(dataset_name, dataset_config, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=kwargs.get("pin_memory", False),
        **{k: v for k, v in kwargs.items() if k not in ("pin_memory",)},
    )
