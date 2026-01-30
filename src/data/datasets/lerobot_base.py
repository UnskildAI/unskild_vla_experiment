"""LeRobot-style dataset adapter: load from HF repo_id or local path (v3 layout)."""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger
from src.core.exceptions import VLADataError

logger = get_logger(__name__)

_LEROBOT_DATASET: Any = None
_LEROBOT_METADATA: Any = None


def _import_lerobot() -> tuple[Any, Any]:
    """Import LeRobotDataset and LeRobotDatasetMetadata; raise if not installed."""
    global _LEROBOT_DATASET, _LEROBOT_METADATA
    if _LEROBOT_DATASET is not None:
        return _LEROBOT_DATASET, _LEROBOT_METADATA
    try:
        try:
            # Newer Lerobot versions (>=0.2)
            from lerobot.datasets.lerobot_dataset import (
                LeRobotDataset,
                LeRobotDatasetMetadata,
            )
        except ImportError:
            # Older Lerobot versions
            from lerobot.common.datasets.lerobot_dataset import (
                LeRobotDataset,
                LeRobotDatasetMetadata,
            )
        _LEROBOT_DATASET = LeRobotDataset
        _LEROBOT_METADATA = LeRobotDatasetMetadata
        return LeRobotDataset, LeRobotDatasetMetadata
    except ImportError as e:
        raise VLADataError(
            "LeRobot dataset support requires the 'lerobot' package. "
            "Install with: pip install lerobot"
        ) from e


def get_lerobot_dataset(
    repo_id: str,
    *,
    root: str | None = None,
    episodes: list[int] | None = None,
    delta_timestamps: dict[str, list[float]] | None = None,
    image_transforms: Any = None,
    revision: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Load a LeRobot-style dataset from HuggingFace Hub or local path.

    - repo_id: HuggingFace repo (e.g. "lerobot/aloha_mobile_cabinet") or dataset name when using root.
    - root: Optional local directory with v3 layout (meta/, data/, videos/). If set, data is loaded from root.
    - episodes: Optional list of episode indices to load.
    - delta_timestamps: Optional temporal windows per key (see LeRobot docs).
    - image_transforms: Optional torchvision or LeRobot ImageTransforms.
    - revision: Optional Hugging Face repo revision (e.g. "main").

    Returns the underlying LeRobotDataset instance (PyTorch Dataset of dicts).
    """
    LeRobotDataset, _ = _import_lerobot()
    load_kwargs: dict[str, Any] = {"repo_id": repo_id, "revision": revision, **kwargs}
    if root is not None:
        load_kwargs["root"] = root
    if episodes is not None:
        load_kwargs["episodes"] = episodes
    if delta_timestamps is not None:
        load_kwargs["delta_timestamps"] = delta_timestamps
    if image_transforms is not None:
        load_kwargs["image_transforms"] = image_transforms
    return LeRobotDataset(**load_kwargs)


def get_lerobot_metadata(repo_id: str, *, root: str | None = None, revision: str | None = None, **kwargs: Any) -> Any:
    """Load only metadata (no data). Useful for camera_keys, fps, total_frames, tasks."""
    _, LeRobotDatasetMetadata = _import_lerobot()
    load_kwargs: dict[str, Any] = {"repo_id": repo_id, "revision": revision, **kwargs}
    if root is not None:
        load_kwargs["root"] = root
    return LeRobotDatasetMetadata(**load_kwargs)
