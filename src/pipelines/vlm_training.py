"""VLM training pipeline: dataset -> trainer -> checkpoint."""

from __future__ import annotations

from typing import Any

from src.pipelines.base import BasePipeline, BasePipelineConfig
from src.config.schema import ExperimentConfig, load_config
from src.data.dataloaders import build_dataloader
from src.core.registry import ModelRegistry, DatasetRegistry
from src.training.trainers.vlm import VLMTrainer, VLMTrainerConfig
from src.core.logging import get_logger


class VLMTrainingPipeline(BasePipeline):
    """VLM finetuning pipeline: config -> dataloader + model + trainer -> train."""

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

    def run(self, **kwargs: Any) -> Any:
        if self.experiment_config is None:
            raise ValueError("Set experiment_config or pass ExperimentConfig to __init__")
        cfg = self.experiment_config
        train_loader = build_dataloader(
            cfg.data.dataset_name,
            cfg.data.extra,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=cfg.data.shuffle,
            drop_last=cfg.data.drop_last,
        )
        model = ModelRegistry.build(cfg.model.name, cfg.model.extra)
        trainer_config = VLMTrainerConfig(
            max_steps=cfg.training.max_steps,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            mixed_precision=cfg.training.mixed_precision,
            log_interval=cfg.training.log_interval,
            eval_interval=cfg.training.eval_interval,
            checkpoint_interval=cfg.training.checkpoint_interval,
            output_dir=cfg.output_dir,
            seed=cfg.training.seed,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            max_grad_norm=cfg.training.max_grad_norm,
        )
        trainer = VLMTrainer(model=model, config=trainer_config)
        trainer.train(train_loader, eval_loader=None)
        return trainer
