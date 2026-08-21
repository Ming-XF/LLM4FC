# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LLM4FC is a research codebase for EEG-based brain analysis using **Time-LLM-style reprogramming**: dynamic functional connectivity (DFC) is extracted from EEG, encoded by a GCN, then fed into a **frozen LLM** (ChatGLM-6B or DeepSeek-R1-Distill-Llama-8B) via cross-attention "reprogramming", followed by a linear classification/regression head. It also ships several baseline graph models (BNT, BrainNetCNN, ALTER, GCDGCN) under the same training harness.

Tasks are either **classification** (disease: AD/MCI/SCD, seizure, abnormal EEG; gender) or **regression** (age prediction). The active research focus (per git history) is the TimeLLM model with LoRA/GC-LoRA and CVIB (conditional variational information bottleneck) extensions.

There is no test suite, linter, or dependency manifest (`requirements.txt`/`pyproject.toml`). Development is driven by running `main.py` directly.

## Commands

Training is invoked through `main.py`, which is typically launched with DeepSpeed via the scripts in `scripts/`:

```bash
# Single model, via script (args: <DATASET> <EARLY_STOP_METRIC> <EARLY_STOP_MIN_DELTA>)
./scripts/train_TimeLLM.sh DiseaseBeirut AUC 0.001

# Direct DeepSpeed launch (equivalent)
deepspeed --num_gpus=6 main.py --model TimeLLM --dataset DiseaseBeirut \
    --deepspeed --do_train --do_evaluate --do_test --num_epochs 200

# Single-GPU / CPU (no distributed): omit --deepspeed/--do_parallel
python main.py --model BNT --dataset DiseaseBeirut --do_train --do_test
```

Key runtime flags (defined in `config.py:init_config`):
- `--model` selects both the model and the trainer via string eval: `eval(args.model + 'Trainer')` in `main.py` and `eval(f"{args.dataset}Dataset")` in `utils/trainer.py`. Valid models: `TimeLLM`, `BNT`, `BrainNetCNN`, `ALTER`, `GCDGCN`.
- `--deepspeed` uses `deepspeed/train.json` (training) or `deepspeed/finetune.json` (few-shot); the DeepSpeed JSON is the **single source of truth for batch size and LR** — `main.py:_apply_deepspeed_config` overwrites `args.batch_size`/`args.learning_rate` from it.
- `--do_parallel` uses `torch.nn.parallel.DistributedDataParallel` instead (init via `torchrun`/env vars).
- `--few_shot 0` = use all data; `>0` = N subjects per class; `-1` = full-data finetune. Combined with `--pretrain_path` this triggers transfer-learning mode (zero-shot / few-shot) in `main.py:main`.
- `--early_stop_metric`/`--early_stop_patience`/`--early_stop_min_delta` control val-based early stopping.

## Architecture

The control flow is: `main.py:main` → `config.py:init_config` (argparse) → instantiate a `Trainer` subclass (`trainers.py`) → `DataConfig` + dataset → `config.py:init_model_config` builds the model → train/eval loop in `utils/trainer.py`.

- **`utils/trainer.py`** — base `Trainer` class holds all training/evaluation logic: `train()`, `finetune()` (few-shot transfer), `evaluate()` (dispatches to `binary_evaluate` / `multiple_evaluate` / `regression_evaluate`), early stopping, `save_model()`/`load_model()`, and DeepSpeed/DDP/AMP dispatch (`_forward`, `_backward_and_step`). Subclasses in `trainers.py` mostly just override `prepare_inputs_kwargs(inputs)` to rename dataset fields for the model's `forward`.
- **`config.py`** — two responsibilities: `init_config()` defines *all* CLI args, and `init_model_config(args, data_config)` is a big `if args.model == ...` factory that constructs each model's `*Config` + `Model`.
- **`model/`** — one subdirectory per model. Each exports `Model` and a `<Name>Config` (subclass of `BaseConfig`). `model/base/` defines `BaseModel`, `BaseConfig`, and `ModelOutputs`. `model/__init__.py` imports everything so `from model import *` works.
- **`data/`** — `data_config.py` (`DataConfig`), `dataset.py` (`BaseDataset`), `dataloader.py` (three loaders: `init_StratifiedKFold_dataloader`, `init_distributed_dataloader`, `init_deepspeed_dataloader`), plus per-dataset modules (`beirut/`, `caueeg/`, `ds/`, `tuab/`, `tuep/`, `multidomain/`). Every dataset class is named `<DatasetName>Dataset` and registered in `data/__init__.py` so `eval(f"{args.dataset}Dataset")` resolves.
- **`utils/`** — `optimizer.py`, `schedule.py`, `recorder.py`, `early_stopping.py`, `logger.py`.

### TimeLLM pipeline (`model/TimeLLM/TimeLLM.py`, the active model)

1. `DFC` (B, T, C, C) is flattened to (B·T, C, C) and passed through a **shared per-window GCN** (`GCNLayer`: linear + `bmm(adj_norm, x)`) using one-hot channel embeddings as node init.
2. Output is reshaped into a patch sequence (B, T·C, d_model) — ordering controlled by `token_order` (`time_first` default / `node_first`).
3. Optional **CVIB** (`_cvib_encode`, gated by `--use_cvib`): `vae` mode adds subject-level KL vs. a label-conditioned prior; `contrastive` mode adds a cosine-alignment loss. Both return an auxiliary loss scaled by `cvib_beta`.
4. `ReprogrammingLayer` cross-attends the patches against learnable text prototypes (word embeddings → `mapping_layer` → num_prototypes).
5. Output is concatenated with prompt embeddings and run through the **frozen LLM**; only patch positions are projected by `output_projection` to logits.

Supporting modules: `model/TimeLLM/gc_lora.py` (LoRA + graph-conditioned GC-LoRA adapters injected into the LLM), `model/TimeLLM/prompts.py` (per-dataset text prompts, keyed by dataset name, fallback to CAUEEG).

## Key conventions

- **Loss is computed inside `forward`.** Models return `ModelOutputs(logits=..., loss=..., hidden_state=...)`, not just logits; the trainer reads `outputs.loss` directly. Classification uses `cross_entropy`, regression uses `huber_loss(delta=1.0)`.
- **Task type is set by each dataset's `load_data`**, which writes `data_config.task_type` and `data_config.output_dim` (1 → regression, ≥2 → classification). `DataConfig.is_classification`/`is_regression` are properties. Regression metrics are denormalized back to real units (years) using `dataset.label_min`/`label_max`.
- **Label format**: binary classification labels are one-hot `(B, 2)`; multi-class are class indices; regression are floats. `prepare_inputs_kwargs` and `_normalize_label` handle the conversions.
- **Subject-grouped splits** prevent leakage. `BaseDataset._create_splits` uses `GroupKFold` when `num_repeat ≥ 2`, otherwise a 60/20/20 subject-level train/val/test split (fixed `random_state=42`). Few-shot samples subjects (not windows) per class via `episode_seed`.
- **Checkpoints** are saved under `--model_dir` (default `output_dir/`) as `{model}-{task_id}.bin` (trainable params + buffers only) plus `config.json`. `load_model` **skips prompt-buffer keys** (`dataset_prompt_embeddings`, `task_prompt_embeddings`) so cross-dataset transfer keeps the target dataset's prompts.
- **Mixed precision**: non-DeepSpeed runs use `torch.cuda.amp` (`GradScaler`); the LLM and its outputs are kept in bfloat16 (see the dtype conversions in `TimeLLM.py`).
- **Reproducibility**: `main.py` hardcodes `set_seed(42)` and `torch.backends.cudnn.deterministic = True`; dataset splits also use seed 42. Few-shot episodes use `--few_shot_seed`.

## Data & model asset locations

- Raw datasets live **outside the repo** at `/root/autodl-tmp/data/` and are loaded via relative paths like `np.load("../data/Beirut/beirut_disease.npy")` (relative to the repo root `/root/autodl-tmp/LLM4FC`). Precomputed `.npy` files include `timeseries`, `labels`, `subject_id`, and precomputed `split_train_index`/`split_val_index`/`split_test_index`.
- LLM checkpoints are committed under `model/chatglm-6b/` and `model/deepseek-r1-distill-llama-8B/` (weights gitignored via `.gitignore`), selected by `--llm_type` / `--llm_path`.
- Sibling reference repos exist at `/root/autodl-tmp/ChatGLM-6B/` and `/root/autodl-tmp/Time-LLM/` (upstream sources this project is derived from).
- `log_dir/` holds training/few-shot/test logs; `output_dir/` holds saved checkpoints and `results.json`.
