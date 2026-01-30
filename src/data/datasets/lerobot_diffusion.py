"""LeRobot dataset view for flow-matching / diffusion: image (+ optional conditioning)."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, Field

from src.data.datasets.base import BaseDataset, BaseDatasetConfig
from src.data.datasets.lerobot_base import get_lerobot_dataset
from src.core.typing import BatchDict
from src.core.logging import get_logger

logger = get_logger(__name__)


class LeRobotDiffusionConfig(BaseDatasetConfig):
    """Config for LeRobot diffusion view (image target; optional text/action conditioning)."""

    repo_id: str = Field(..., description="HuggingFace repo_id or dataset name")
    root: str | None = Field(None, description="Local path to v3 dataset")
    camera_key: str | None = Field(None, description="Image key; default: first camera")
    image_size: int = Field(64, gt=0)
    conditioning: str = Field("none", description="none, text, action")
    episodes: list[int] | None = Field(None, description="Subset of episode indices")
    revision: str | None = Field(None, description="Hugging Face repo revision (e.g. 'main')")
    extra: dict[str, Any] = Field(default_factory=dict)


class LeRobotDiffusionDataset(BaseDataset):
    """
    LeRobot dataset as image target for flow-matching / diffusion.
    Optional conditioning: task text or action.
    Outputs: image; conditioning_text / conditioning_action when enabled.
    """

    def __init__(self, config: LeRobotDiffusionConfig | dict[str, Any], **kwargs: Any) -> None:
        if isinstance(config, dict):
            config = LeRobotDiffusionConfig(**config)
        super().__init__(config)
        self.image_size = config.image_size
        self.conditioning = config.conditioning
        self._ds = get_lerobot_dataset(
            config.repo_id,
            root=config.root,
            episodes=config.episodes,
            revision=config.revision,
            **kwargs,
        )
        meta = getattr(self._ds, "meta", None)
        camera_keys = getattr(meta, "camera_keys", None)
        if camera_keys is None:
            camera_keys = []
        self._camera_key = config.camera_key or (camera_keys[0] if len(camera_keys) > 0 else "observation.images.top")
        tasks = getattr(meta, "tasks", None)
        self._tasks = tasks if tasks is not None else {}

    def __len__(self) -> int:
        return len(self._ds)

    def _get_task_text(self, index: int) -> str:
        raw = self._ds[index]
        idx = raw.get("task_index")
        if idx is not None and isinstance(idx, torch.Tensor):
            idx = idx.item() if idx.numel() == 1 else int(idx.flatten()[0])
        if idx is not None and isinstance(self._tasks, (list, dict)):
            if isinstance(self._tasks, dict):
                return self._tasks.get(str(idx), self._tasks.get(idx, "")) or ""
            if isinstance(self._tasks, list) and 0 <= idx < len(self._tasks):
                return self._tasks[idx] if isinstance(self._tasks[idx], str) else str(self._tasks[idx])
        return ""

    def __getitem__(self, index: int) -> BatchDict:
        raw = self._ds[index]
        image = raw.get(self._camera_key)
        if image is None:
            image = torch.rand(3, self.image_size, self.image_size)
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        if image.dim() == 4:
            image = image[0]
        if image.shape[0] != 3 or image.shape[1] != self.image_size or image.shape[2] != self.image_size:
            from torchvision.transforms import functional as F
            image = F.resize(image.unsqueeze(0), [self.image_size, self.image_size]).squeeze(0)
            if image.shape[0] != 3:
                image = image.expand(3, -1, -1)
        out: BatchDict = {"image": image.float()}
        if self.conditioning == "text":
            out["conditioning_text"] = self._get_task_text(index)
        if self.conditioning == "action":
            action = raw.get("action")
            if action is not None and isinstance(action, torch.Tensor):
                out["conditioning_action"] = action.flatten().float()
            else:
                out["conditioning_action"] = torch.zeros(1)
        return out
