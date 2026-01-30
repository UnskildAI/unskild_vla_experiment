"""LeRobot dataset view for VLM finetuning: image + task text per frame."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, Field

from src.data.datasets.base import BaseDataset, BaseDatasetConfig
from src.data.datasets.lerobot_base import get_lerobot_dataset
from src.core.typing import BatchDict
from src.core.logging import get_logger

logger = get_logger(__name__)


class LeRobotVLMConfig(BaseDatasetConfig):
    """Config for LeRobot VLM view (image + task text)."""

    repo_id: str = Field(..., description="HuggingFace repo_id or dataset name")
    root: str | None = Field(None, description="Local path to v3 dataset (meta/, data/, videos/)")
    camera_key: str | None = Field(None, description="Image key, e.g. observation.images.top; default: first camera")
    max_length: int = Field(512, gt=0)
    image_size: int = Field(224, gt=0)
    single_task: str | None = Field(None, description="Override task string for all samples")
    episodes: list[int] | None = Field(None, description="Subset of episode indices")
    revision: str | None = Field(None, description="Hugging Face repo revision (e.g. 'main')")
    extra: dict[str, Any] = Field(default_factory=dict)


class LeRobotVLMDataset(BaseDataset):
    """
    LeRobot dataset as image + task text for VLM finetuning.
    Each sample: one frame image + task description (from meta/tasks or single_task).
    Outputs: pixel_values, input_ids, attention_mask, labels (tokenizer optional).
    """

    def __init__(self, config: LeRobotVLMConfig | dict[str, Any], tokenizer: Any = None, **kwargs: Any) -> None:
        if isinstance(config, dict):
            config = LeRobotVLMConfig(**config)
        super().__init__(config)
        self.max_length = config.max_length
        self.image_size = config.image_size
        self.single_task = config.single_task
        self._tokenizer = tokenizer
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
        features = getattr(self._ds, "features", None)
        self._task_key = "task_index" if features is not None and "task_index" in features else None

    def __len__(self) -> int:
        return len(self._ds)

    def _get_task_text(self, index: int) -> str:
        if self.single_task:
            return self.single_task
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
        task_text = self._get_task_text(index)
        if self._tokenizer is not None:
            tok = self._tokenizer(
                task_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].squeeze(0)
            attention_mask = tok["attention_mask"].squeeze(0)
            labels = input_ids.clone()
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.ones(self.max_length, dtype=torch.long)
            if task_text:
                for i, c in enumerate(task_text[: self.max_length - 1]):
                    input_ids[i] = ord(c) % 32000
            labels = input_ids.clone()
        return {
            "pixel_values": image.float(),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
