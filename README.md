# Vision-Language-Action (VLA) Training & Finetuning Framework

Modular training and finetuning infrastructure for VLMs (PaliGemma, LLaVA, Idefics), diffusion/flow-matching models, and vision → action policies. Component-wise finetuning (VLM, vision encoder, language head, action head, diffusion) is supported; all base classes are abstract, torch-native, and accept Pydantic configs.

## Requirements

- Python 3.11+
- PyTorch 2.0+
- uv or pip

## Install

```bash
uv sync
# or: pip install -e .
```

## Project layout

- **`src/core/`** — Registry, typing, exceptions, logging
- **`src/data/`** — BaseDataset, VLM/action/diffusion datasets, transforms, dataloaders
- **`src/models/`** — BaseVLM, BaseDiffusionModel, BaseActionHead; PaliGemma, LLaVA, flow-matching, MLP/diffusion action heads
- **`src/training/`** — BaseTrainer, VLM/diffusion/action trainers, losses, optimizers, schedulers, callbacks
- **`src/evaluation/`** — Vision/language/action metrics, benchmarks
- **`src/pipelines/`** — BasePipeline, VLM/diffusion/action training pipelines, hybrid
- **`src/config/`** — Pydantic schema, YAML load with env expansion
- **`src/cli/`** — `train`, `eval`, `export` entrypoints
- **`config/`** — Example YAMLs (model, data, training, experiments)

## CLI

After install, entrypoints: `vla-train`, `vla-eval`, `vla-export`.

**Train (from experiment YAML):**

```bash
vla-train config/experiments/vlm_finetune.yaml --pipeline vlm
vla-train config/experiments/flow_matching.yaml --pipeline diffusion
vla-train config/experiments/action_policy.yaml --pipeline action
```

**Eval:**

```bash
vla-eval config/experiments/vlm_finetune.yaml path/to/ckpt.pt
```

**Export (state_dict):**

```bash
vla-export config/experiments/vlm_finetune.yaml --output model.pt --format torchscript
```

## Design

- **Separable finetuning:** Vision encoder, language decoder, action head, diffusion model can be frozen or swapped independently.
- **Base abstractions:** `BaseDataset`, `BaseVLM`, `BaseDiffusionModel`, `BaseActionHead`, `BaseTrainer`, `BasePipeline` are ABCs with Pydantic config and checkpoint save/load.
- **Config:** YAML-based, env var expansion, no logic in config files; experiment reproducibility via seed, model hash, dataset version.
- **Training:** Explicit training loops, gradient accumulation, mixed precision, optional distributed (accelerate); no magic helpers.
- **Diffusion/flow matching:** Explicit noise/flow schedules, continuous time, pluggable loss, dataset-level conditioning (image, text, action).

## Example configs

- **VLM finetuning:** `config/experiments/vlm_finetune.yaml`
- **Flow-matching diffusion:** `config/experiments/flow_matching.yaml`
- **Vision → action policy:** `config/experiments/action_policy.yaml`
- **LeRobot VLM / action / diffusion:** `config/experiments/lerobot_vlm_finetune.yaml`, `lerobot_action_finetune.yaml`, `lerobot_diffusion_finetune.yaml`

No hard-coded model assumptions; models and datasets are registered and built from config.

## Complete procedure: load from Hugging Face and finetune

**Step-by-step guide:** [docs/LOAD_AND_FINETUNE.md](docs/LOAD_AND_FINETUNE.md)

Covers: install + LeRobot, HF login, choosing a dataset, loading it via config or `--repo-id`, and running each pipeline. Minimal copy-paste:

```bash
uv sync && uv sync --extra lerobot
vla-train config/experiments/lerobot_vlm_finetune.yaml --pipeline vlm --repo-id lerobot/aloha_mobile_cabinet
vla-train config/experiments/lerobot_action_finetune.yaml --pipeline action --repo-id lerobot/aloha_mobile_cabinet
vla-train config/experiments/lerobot_diffusion_finetune.yaml --pipeline diffusion --repo-id lerobot/aloha_mobile_cabinet
```

## LeRobot-style datasets

You can finetune **VLM**, **vision→action**, and **diffusion** models individually on a [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)–style dataset (HuggingFace Hub or local `meta/`, `data/`, `videos/` layout).

**Install LeRobot (optional):**

```bash
pip install lerobot
# or: uv sync --extra lerobot
```

**Dataset views (same dataset, different training targets):**

| Dataset name        | Use case              | Outputs                                      |
|---------------------|------------------------|----------------------------------------------|
| `lerobot_vlm`       | VLM finetuning         | `pixel_values`, `input_ids`, `attention_mask`, `labels` (image + task text) |
| `lerobot_action`     | Vision→action policy   | `pixel_values`, `action`, `action_mask`     |
| `lerobot_diffusion` | Flow-matching / diffusion | `image`; optional `conditioning_text` / `conditioning_action` |

**Config (YAML):** Set `data.dataset_name` to one of the above and in `data.extra` set:

- **`repo_id`**: HuggingFace repo (e.g. `lerobot/aloha_mobile_cabinet`) or dataset name.
- **`root`** (optional): Local path to a v3 layout (enables loading without Hub).
- **`camera_key`** (optional): Image key (default: first camera from metadata).
- **`image_size`**, **`max_length`** (VLM), **`action_chunk_size`** (action), **`conditioning`** (diffusion), etc. as needed.

**Example: finetune each model type on the same LeRobot dataset:**

```bash
# With repo_id in YAML: edit data.extra.repo_id in the config, then:
vla-train config/experiments/lerobot_vlm_finetune.yaml --pipeline vlm
vla-train config/experiments/lerobot_action_finetune.yaml --pipeline action
vla-train config/experiments/lerobot_diffusion_finetune.yaml --pipeline diffusion

# Or override from CLI (no YAML edit):
vla-train config/experiments/lerobot_vlm_finetune.yaml --pipeline vlm --repo-id lerobot/aloha_mobile_cabinet
```

Override `data.extra.repo_id` in the YAML, via `--repo-id`, or via env (e.g. `repo_id: ${HF_DATASET_REPO}`).
