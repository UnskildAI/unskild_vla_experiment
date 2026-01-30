"""CLI entrypoint: train from YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.schema import load_config
from src.core.logging import setup_logging
import src.data.datasets  # noqa: F401
import src.models  # noqa: F401
from src.pipelines.vlm_training import VLMTrainingPipeline
from src.pipelines.diffusion_training import DiffusionTrainingPipeline
from src.pipelines.action_training import ActionTrainingPipeline

PIPELINES = {
    "vlm": VLMTrainingPipeline,
    "diffusion": DiffusionTrainingPipeline,
    "action": ActionTrainingPipeline,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train from YAML config. Optionally override Hugging Face dataset via --repo-id."
    )
    parser.add_argument("config", type=Path, help="Path to experiment YAML")
    parser.add_argument("--pipeline", choices=list(PIPELINES), default="vlm", help="Pipeline type")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face dataset repo_id (e.g. lerobot/aloha_mobile_cabinet). Overrides data.extra.repo_id.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Hugging Face dataset revision (e.g. main). Passed to LeRobotDataset.",
    )
    parser.add_argument("--no-expand-env", action="store_true", help="Disable env var expansion")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config, expand_env=not args.no_expand_env)
    if args.repo_id is not None:
        if not hasattr(cfg.data, "extra") or cfg.data.extra is None:
            cfg.data.extra = {}
        cfg.data.extra["repo_id"] = args.repo_id
    if args.revision is not None:
        if not hasattr(cfg.data, "extra") or cfg.data.extra is None:
            cfg.data.extra = {}
        cfg.data.extra["revision"] = args.revision
    pipeline_cls = PIPELINES[args.pipeline]
    pipeline = pipeline_cls(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
