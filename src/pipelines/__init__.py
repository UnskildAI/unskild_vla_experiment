"""Training pipelines: compose stages for VLM, diffusion, action."""

from src.pipelines.base import BasePipeline
from src.pipelines.vlm_training import VLMTrainingPipeline
from src.pipelines.diffusion_training import DiffusionTrainingPipeline
from src.pipelines.action_training import ActionTrainingPipeline
from src.pipelines.hybrid import HybridPipeline

__all__ = [
    "BasePipeline",
    "VLMTrainingPipeline",
    "DiffusionTrainingPipeline",
    "ActionTrainingPipeline",
    "HybridPipeline",
]
