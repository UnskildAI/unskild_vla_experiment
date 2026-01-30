"""Hybrid pipeline: optional VLM + diffusion + action stages (compose)."""

from __future__ import annotations

from typing import Any

from src.pipelines.base import BasePipeline, BasePipelineConfig
from src.config.schema import ExperimentConfig
from src.core.logging import get_logger


class HybridPipeline(BasePipeline):
    """Optional composition: VLM finetuning -> diffusion -> action. Stages run in order if enabled."""

    def __init__(self, config: BasePipelineConfig | ExperimentConfig, **kwargs: Any) -> None:
        if isinstance(config, ExperimentConfig):
            self.experiment_config = config
            config = BasePipelineConfig(
                experiment_name=config.experiment_name,
                output_dir=config.output_dir,
                seed=config.seed,
            )
        else:
            self.experiment_config = None
        super().__init__(config)
        self.logger = get_logger(self.__class__.__name__)

    def run(self, vlm: bool = False, diffusion: bool = False, action: bool = False, **kwargs: Any) -> Any:
        if self.experiment_config is None:
            raise ValueError("Set experiment_config or pass ExperimentConfig to __init__")
        results = []
        if vlm:
            from src.pipelines.vlm_training import VLMTrainingPipeline
            p = VLMTrainingPipeline(self.experiment_config)
            results.append(("vlm", p.run(**kwargs)))
        if diffusion:
            from src.pipelines.diffusion_training import DiffusionTrainingPipeline
            p = DiffusionTrainingPipeline(self.experiment_config)
            results.append(("diffusion", p.run(**kwargs)))
        if action:
            from src.pipelines.action_training import ActionTrainingPipeline
            p = ActionTrainingPipeline(self.experiment_config)
            results.append(("action", p.run(**kwargs)))
        if not results:
            self.logger.warning("No stages enabled; set vlm=True, diffusion=True, or action=True")
        return dict(results)
