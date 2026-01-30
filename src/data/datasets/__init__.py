"""Dataset implementations."""

from src.core.registry import DatasetRegistry
from src.data.datasets.base import BaseDataset
from src.data.datasets.vlm import VLMDataset, VLMDatasetConfig
from src.data.datasets.action import ActionDataset, ActionDatasetConfig
from src.data.datasets.diffusion import DiffusionDataset, DiffusionDatasetConfig
from src.data.datasets.lerobot_vlm import LeRobotVLMDataset, LeRobotVLMConfig
from src.data.datasets.lerobot_action import LeRobotActionDataset, LeRobotActionConfig
from src.data.datasets.lerobot_diffusion import LeRobotDiffusionDataset, LeRobotDiffusionConfig

DatasetRegistry.register("vlm")(VLMDataset)
DatasetRegistry.register("action")(ActionDataset)
DatasetRegistry.register("diffusion")(DiffusionDataset)
DatasetRegistry.register("lerobot_vlm")(LeRobotVLMDataset)
DatasetRegistry.register("lerobot_action")(LeRobotActionDataset)
DatasetRegistry.register("lerobot_diffusion")(LeRobotDiffusionDataset)

__all__ = [
    "BaseDataset",
    "VLMDataset",
    "VLMDatasetConfig",
    "ActionDataset",
    "ActionDatasetConfig",
    "DiffusionDataset",
    "DiffusionDatasetConfig",
    "LeRobotVLMDataset",
    "LeRobotVLMConfig",
    "LeRobotActionDataset",
    "LeRobotActionConfig",
    "LeRobotDiffusionDataset",
    "LeRobotDiffusionConfig",
]
