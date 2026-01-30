"""CLI entrypoint: export model to ONNX / TorchScript (stub)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.schema import load_config
from src.core.registry import ModelRegistry
from src.core.logging import setup_logging
import src.models  # noqa: F401
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to ONNX / TorchScript")
    parser.add_argument("config", type=Path, help="Path to experiment YAML")
    parser.add_argument("checkpoint", type=Path, nargs="?", default=None, help="Path to checkpoint (optional)")
    parser.add_argument("--output", type=Path, default=Path("exported.pt"), help="Output path")
    parser.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    parser.add_argument("--no-expand-env", action="store_true", help="Disable env var expansion")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config, expand_env=not args.no_expand_env)
    model = ModelRegistry.build(cfg.model.name, cfg.model.extra)
    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state), strict=True)
    model.eval()
    if args.format == "torchscript":
        torch.save(model.state_dict(), args.output)
    else:
        raise NotImplementedError("ONNX export: provide example inputs in code")
    print("Exported to", args.output)


if __name__ == "__main__":
    main()
