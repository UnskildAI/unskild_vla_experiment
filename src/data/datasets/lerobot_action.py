"""LeRobot dataset view for vision -> action policy: image + action per frame."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, Field

from src.data.datasets.base import BaseDataset, BaseDatasetConfig
from src.data.datasets.lerobot_base import get_lerobot_dataset
from src.core.typing import BatchDict
from src.core.logging import get_logger

logger = get_logger(__name__)


class LeRobotActionConfig(BaseDatasetConfig):
    """Config for LeRobot action view (image + action)."""

    repo_id: str = Field(..., description="HuggingFace repo_id or dataset name")
    root: str | None = Field(None, description="Local path to v3 dataset")
    camera_key: str | None = Field(None, description="Image key; default: first camera")
    image_size: int = Field(224, gt=0)
    action_key: str = Field("action", description="Key for action tensor")
    action_chunk_size: int | None = Field(None, description="Use future action chunk; 1 = current only")
    episodes: list[int] | None = Field(None, description="Subset of episode indices")
    revision: str | None = Field(None, description="Hugging Face repo revision (e.g. 'main')")
    extra: dict[str, Any] = Field(default_factory=dict)


class LeRobotActionDataset(BaseDataset):
    """
    LeRobot dataset as vision -> action for policy finetuning.
    Each sample: one frame image + action (current or chunk).
    Outputs: pixel_values, action, action_mask.
    """

    def __init__(self, config: LeRobotActionConfig | dict[str, Any], **kwargs: Any) -> None:
        if isinstance(config, dict):
            config = LeRobotActionConfig(**config)
        super().__init__(config)
        self.image_size = config.image_size
        self.action_key = config.action_key
        self.action_chunk_size = config.action_chunk_size or 1
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

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> BatchDict:
        raw = self._ds[index]
        image = raw.get(self._camera_key)
        if image is None:
            image = torch.zeros(3, self.image_size, self.image_size)
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        if image.dim() == 4:
            image = image[0]
        if image.shape[0] != 3 or image.shape[1] != self.image_size or image.shape[2] != self.image_size:
            from torchvision.transforms import functional as F
            image = F.resize(image.unsqueeze(0), [self.image_size, self.image_size]).squeeze(0)
            if image.shape[0] != 3:
                image = image.expand(3, -1, -1)
        action = raw.get(self.action_key)
        if action is None:
            action = torch.zeros(1)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        if action.dim() > 1:
            action = action.flatten()
        if self.action_chunk_size > 1 and action.numel() >= self.action_chunk_size:
            action = action[: self.action_chunk_size]
        elif self.action_chunk_size > 1:
            action = torch.nn.functional.pad(action, (0, self.action_chunk_size - action.numel()))
        return {
            "pixel_values": image.float(),
            "action": action.float(),
            "action_mask": torch.ones_like(action),
        }
