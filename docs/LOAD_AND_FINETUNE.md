# Complete procedure: load dataset from Hugging Face and finetune

This guide walks through loading a LeRobot-style dataset from the Hugging Face Hub and finetuning **VLM**, **vision→action**, and **diffusion** models individually using the same dataset.

---

## 1. Prerequisites

- **Python 3.11+**, **PyTorch 2.0+**
- **Hugging Face account** (for gated or private datasets)
- **LeRobot-style dataset** on the Hub (v3 layout: `meta/`, `data/`, `videos/`)

### 1.1 Install the project and LeRobot

```bash
# From project root
uv sync
uv sync --extra lerobot
# or: pip install -e ".[lerobot]"
```

LeRobot is required to load datasets from the Hub (or local v3 layout).

### 1.2 (Optional) Log in to Hugging Face

For gated or private datasets:

```bash
pip install huggingface_hub
huggingface-cli login
```

Enter your token when prompted. For public datasets this step is optional.

---

## 2. Choose a dataset on Hugging Face

Use a LeRobot v3 dataset. Examples:

| Repo ID | Description |
|---------|-------------|
| `lerobot/aloha_mobile_cabinet` | ALOHA mobile cabinet (public) |
| `yaak-ai/L2D-v3` | Example v3 port |
| `your-username/your-dataset` | Your own LeRobot v3 dataset |

Browse: [Hugging Face datasets tagged LeRobot](https://huggingface.co/datasets?other=LeRobot).

Set the **repo_id** you will use (e.g. `lerobot/aloha_mobile_cabinet`).

---

## 3. Load dataset and pass it to a pipeline

The framework loads the dataset **from the config**: `data.dataset_name` and `data.extra` (including `repo_id`) are passed to `build_dataloader`, which instantiates the right dataset and wraps it in a DataLoader. Each pipeline (VLM, action, diffusion) then uses that dataloader for training.

You can point to the Hub dataset in either of two ways.

### Option A: Set `repo_id` in the experiment YAML

1. Copy the right base config and set `data.extra.repo_id`:

   **VLM (image + task text):**
   ```yaml
   # config/experiments/my_lerobot_vlm.yaml
   experiment_name: my_lerobot_vlm
   output_dir: ./outputs/my_lerobot_vlm
   seed: 42
   model:
     name: paligemma
     extra: { vision_frozen: false, language_frozen: false }
   data:
     dataset_name: lerobot_vlm
     batch_size: 16
     num_workers: 0
     shuffle: true
     drop_last: true
     extra:
       repo_id: lerobot/aloha_mobile_cabinet   # <-- your HF dataset
       max_length: 512
       image_size: 224
   training:
     max_steps: 5000
     learning_rate: 1.0e-5
     log_interval: 20
     checkpoint_interval: 1000
   ```

   **Action (vision→action):**
   ```yaml
   # config/experiments/my_lerobot_action.yaml
   data:
     dataset_name: lerobot_action
     extra:
       repo_id: lerobot/aloha_mobile_cabinet
       image_size: 224
       action_key: action
       action_chunk_size: 1
   model:
     name: vision_action
     extra: { input_dim: 2048, action_dim: 14 }
   # ... rest like lerobot_action_finetune.yaml
   ```

   **Diffusion (flow-matching on images):**
   ```yaml
   # config/experiments/my_lerobot_diffusion.yaml
   data:
     dataset_name: lerobot_diffusion
     extra:
       repo_id: lerobot/aloha_mobile_cabinet
       image_size: 64
       conditioning: none
   # ... rest like lerobot_diffusion_finetune.yaml
   ```

2. Run the matching pipeline:

   ```bash
   vla-train config/experiments/my_lerobot_vlm.yaml --pipeline vlm
   vla-train config/experiments/my_lerobot_action.yaml --pipeline action
   vla-train config/experiments/my_lerobot_diffusion.yaml --pipeline diffusion
   ```

### Option B: Override `repo_id` from the command line

Use a base LeRobot config and pass `--repo-id` so you don’t edit YAML for each dataset:

```bash
vla-train config/experiments/lerobot_vlm_finetune.yaml --pipeline vlm --repo-id lerobot/aloha_mobile_cabinet
vla-train config/experiments/lerobot_action_finetune.yaml --pipeline action --repo-id lerobot/aloha_mobile_cabinet
vla-train config/experiments/lerobot_diffusion_finetune.yaml --pipeline diffusion --repo-id lerobot/aloha_mobile_cabinet
```

If the base YAML already has a `repo_id` in `data.extra`, `--repo-id` overrides it.

---

## 4. End-to-end flow (what happens when you run `vla-train`)

1. **Load config**  
   `load_config(config_path)` reads the YAML and (if you passed `--repo-id`) overrides `data.extra["repo_id"]`.

2. **Build dataset and DataLoader**  
   The pipeline calls `build_dataloader(cfg.data.dataset_name, cfg.data.extra, batch_size=..., ...)`.  
   - For `lerobot_vlm` / `lerobot_action` / `lerobot_diffusion`, the corresponding class uses `get_lerobot_dataset(repo_id=cfg.data.extra["repo_id"], root=..., episodes=...)` to load from the Hub (or local `root`).  
   - The dataset is wrapped in a PyTorch `DataLoader` and returned.

3. **Build model**  
   The pipeline builds the model from `cfg.model.name` and `cfg.model.extra` (e.g. `paligemma`, `vision_action`, `flow_matching`).

4. **Train**  
   The pipeline creates the trainer (VLM / action / diffusion), passes the dataloader, and runs the training loop (gradient accumulation, mixed precision, checkpointing as in config).

So: **loading from Hugging Face is done inside the dataset class using `repo_id` from config; that dataloader is passed to the chosen pipeline and used for finetuning.**

---

## 5. Minimal copy-paste procedure

Replace `lerobot/aloha_mobile_cabinet` with your dataset repo_id if different.

```bash
# 1. Install
uv sync && uv sync --extra lerobot

# 2. (Optional) HF login for gated/private datasets
huggingface-cli login

# 3. Finetune VLM (image + task text)
vla-train config/experiments/lerobot_vlm_finetune.yaml --pipeline vlm --repo-id lerobot/aloha_mobile_cabinet

# 4. Finetune vision→action policy
vla-train config/experiments/lerobot_action_finetune.yaml --pipeline action --repo-id lerobot/aloha_mobile_cabinet

# 5. Finetune flow-matching diffusion
vla-train config/experiments/lerobot_diffusion_finetune.yaml --pipeline diffusion --repo-id lerobot/aloha_mobile_cabinet
```

Checkpoints and logs go to the `output_dir` in each experiment YAML (e.g. `./outputs/lerobot_vlm_finetune`).

---

## 6. Using a local dataset (v3 layout)

If the dataset is on disk with LeRobot v3 layout (`meta/`, `data/`, `videos/`):

1. In the YAML, set `data.extra.root` to the directory path and keep `repo_id` as a dataset name (e.g. `my_local_dataset`):

   ```yaml
   data:
     dataset_name: lerobot_vlm
     extra:
       repo_id: my_local_dataset
       root: /path/to/lerobot/v3/root
       max_length: 512
       image_size: 224
   ```

2. Run as usual:

   ```bash
   vla-train config/experiments/lerobot_vlm_finetune.yaml --pipeline vlm
   ```

Do **not** pass `--repo-id` when using only a local path; the loader uses `root` and `repo_id` from the config.

---

## 7. Overriding with environment variables

Config supports env expansion. You can keep a single YAML and override the repo:

```yaml
# data.extra
extra:
  repo_id: ${HF_DATASET_REPO}
  max_length: 512
  image_size: 224
```

```bash
export HF_DATASET_REPO=lerobot/aloha_mobile_cabinet
vla-train config/experiments/lerobot_vlm_finetune.yaml --pipeline vlm
```

---

## 8. Summary

| Step | Action |
|------|--------|
| 1 | Install project + `lerobot`; optionally `huggingface-cli login` |
| 2 | Pick a LeRobot v3 dataset on the Hub (repo_id) |
| 3 | Set `data.extra.repo_id` in a LeRobot experiment YAML **or** pass `--repo-id` to `vla-train` |
| 4 | Run `vla-train <config> --pipeline vlm|action|diffusion` |
| 5 | Dataset is loaded from HF inside the dataset class; the same dataset can drive all three pipelines by switching `dataset_name` and pipeline |

The dataset is **always** passed to the pipeline indirectly: the pipeline builds a DataLoader from `data.dataset_name` and `data.extra` (including `repo_id`), and uses that DataLoader for training.
