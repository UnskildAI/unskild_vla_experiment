"""CLI entrypoint: evaluate from config and checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.schema import load_config
from src.core.registry import ModelRegistry
import src.data.datasets  # noqa: F401
import src.models  # noqa: F401
from src.data.dataloaders import build_dataloader
from src.evaluation.benchmarks import run_benchmark
from src.evaluation.metrics.vision import vision_metrics
from src.evaluation.metrics.language import language_metrics
from src.evaluation.metrics.action import action_metrics
from src.core.logging import setup_logging
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate from config and checkpoint")
    parser.add_argument("config", type=Path, help="Path to experiment YAML")
    parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint")
    parser.add_argument("--no-expand-env", action="store_true", help="Disable env var expansion")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config, expand_env=not args.no_expand_env)
    model = ModelRegistry.build(cfg.model.name, cfg.model.extra)
    state = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    dataloader = build_dataloader(
        cfg.data.dataset_name,
        cfg.data.extra,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )
    def _vision(out, batch):
        pred = out.get("logits") or out.get("image") or batch.get("pixel_values")
        tgt = batch.get("pixel_values") or batch.get("image")
        if pred is not None and tgt is not None:
            return vision_metrics(pred, tgt)
        return {}

    def _language(out, batch):
        logits = out.get("logits")
        labels = batch.get("labels") or batch.get("input_ids")
        if logits is not None and labels is not None:
            return language_metrics(logits, labels)
        return {}

    def _action(out, batch):
        pred = out.get("action") or out.get("logits") if isinstance(out, dict) else out
        tgt = batch.get("action")
        if pred is not None and tgt is not None:
            return action_metrics(pred, tgt)
        return {}

    metric_fns = {"vision": _vision, "language": _language, "action": _action}
    result = run_benchmark(model, dataloader, metric_fns)
    print("Eval result:", result)


if __name__ == "__main__":
    main()
