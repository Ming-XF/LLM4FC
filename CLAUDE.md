# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

LLM4FC applies frozen LLMs (ChatGLM-6B, DeepSeek-R1-Distill-Llama-8B) to EEG-based functional connectivity analysis. Dynamic functional connectivity (DFC) matrices are encoded via GCN, "reprogrammed" into language-token embeddings, and fed through the LLM for prediction. Tasks: disease classification, age regression, gender classification, and future FC prediction across five EEG datasets (CAUEEG, DS, TUAB, TUEP, Beirut).

## Commands

### Training

Training uses DeepSpeed shell wrappers in `scripts/`. All require 6 GPUs:

```bash
# TimeLLM (transfer learning from a CAUEEG disease pretrain)
./scripts/train_TimeLLM.sh <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>
# e.g. ./scripts/train_TimeLLM.sh DiseaseTUAB ../data/TUAB/tuab_disease.npz AUC

# Baseline models
./scripts/train_BNT.sh <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>
./scripts/train_BrainNetCNN.sh <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>
./scripts/train_ALTER.sh <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>
./scripts/train_GCDGCN.sh <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>
```

The `DATASET` argument follows the pattern `{Task}{Source}Dataset`, e.g. `DiseaseCAUEEGDataset`, `AgeTUABDataset`, `FutureFCTUEPDataset`. These are class names in `data/__init__.py`.

Direct `deepspeed` invocation for custom runs:

```bash
deepspeed --num_gpus=6 main.py \
    --model TimeLLM --dataset DiseaseCAUEEGDataset --data_dir ../data/CAUEEG/caueeg_disease.npz \
    --llm_type llama --llm_path ./model/deepseek-r1-distill-llama-8B \
    --use_dataset_prompt --use_task_prompt --use_stats_prompt \
    --batch_size 2 --num_epochs 200 --deepspeed --do_train --do_evaluate --do_test
```

### Preprocessing

Each dataset module runs directly to produce the `.npz` file consumed by training:

```bash
python data/caueeg/disease_caueeg.py
python data/caueeg/futurefc_caueeg.py
# etc.
```

Preprocessing reads raw EDF files, computes window caps per subject, and writes a `.npz` with precomputed subject split indices. Parameters like `max_windows_per_subject` and `max_subjects` are set in the `__main__` block of each preprocessing file — see `.claude/skills/find-preprocess-params.md` for the workflow to find optimal values.

### No test suite

There is no test suite, linter config, or CI in this repository.

## Architecture

### Reflection-based dispatch

Models, datasets, and trainers are wired by string names. `main.py` instantiates the trainer with `eval(args.model + 'Trainer')` and datasets via `eval(f"{args.dataset}Dataset")`. This means:

- Each model needs a `{ModelName}Trainer` class in `trainers.py`
- Each dataset needs a `{Task}{Source}Dataset` class imported in `data/__init__.py`
- Early stop metric must match an evaluation metric name (AUC, Accuracy, Sensitivity, Specificity, Loss for regression)

### Three compute backends

The `Trainer` base class (`utils/trainer.py`) isolates three backends selected by CLI flags:

| Flag | Backend | Loader |
|---|---|---|
| `--deepspeed` | DeepSpeed ZeRO-2 | `init_deepspeed_dataloader` |
| `--do_parallel` | PyTorch DDP | `init_distributed_dataloader` |
| (neither) | Single-GPU AMP | `init_StratifiedKFold_dataloader` |

The backend affects `__init__` (engine setup), `_forward`, `_backward_and_step`, dataloader selection, and model save/load (DeepSpeed requires gathering ZeRO-sharded params before save).

### Data flow

1. `BaseDataset` (`data/dataset.py`) computes static FC (Pearson correlation across all channels) and dynamic FC (sliding windows) from preprocessed `.npz` files, applies optional sparsification (`--fc_threshold`, `--fc_keep_ratio`), and handles train/val/test splits. Splits are **by subject** (60/20/20) to prevent data leakage, or via `GroupKFold` when `num_repeat > 1`.
2. `DataConfig` (`data/data_config.py`) holds metadata (node_size, output_dim, task_type). `task_type` is set by each dataset's `load_data` and determines which evaluation function runs (classification → AUC/Accuracy, regression → MSE/MAE/RMSE/R², multi-output regression → per-element PearsonR).
3. `Trainer.prepare_inputs_kwargs` (in `trainers.py`) maps dataset batch dict → model inputs. `TimeLLMTrainer` handles all three task types with different label formats.

### TimeLLM model (`model/TimeLLM/TimeLLM.py`)

Key components in order of data flow:
- **GCNLayer**: projects DFC matrices to node embeddings via `torch.bmm` with normalized adjacency
- **ReprogrammingLayer**: cross-attention that maps GCN embeddings into LLM token-embedding space using a learned "source embedding"
- **Prompt construction**: pre-tokenized dataset/task/stats prompt embeddings are registered as buffers and concatenated with reprogrammed tokens before the LLM forward pass. Per-sample FC statistics prompts are built dynamically from the static FC matrix.
- **Frozen LLM**: ChatGLM or Llama backbone (`requires_grad=False`, bf16). Position IDs and causal masks branch by `llm_type` (ChatGLM uses 2D positional encoding, Llama uses RoPE).
- **Output head**: linear over flattened LLM output for classification/regression; a differentiable Pearson FC head (`_pearson_fc_head`) for FutureFC multi-output regression.

Prompts per dataset are defined in `model/TimeLLM/prompts.py` — hand-written descriptions of dataset, task, and stats templates for every combination of dataset × task.

### Baseline models

- **BNT** (`model/BrainNetworkTransformer/`): transformer with deep-clustering auxiliary decoder
- **BrainNetCNN** (`model/BrainNetCNN/`): edge-to-edge CNN over ROI adjacency
- **ALTER** (`model/ALTER/`): transformer-encoder with RRWP positional encodings and clustering
- **GCDGCN** (`model/GCDGCN/`): multi-layer GCN with spectral pooling. Has a two-phase training: pretrain (first 100 epochs) then finetune.

### Transfer learning

When `--pretrain_path` is set, `main.py` enters transfer mode: loads a pretrained checkpoint, optionally samples K subjects per class (`--few_shot`, `--few_shot_seed`), fine-tunes with `Trainer.finetune()`, then tests. Prompt-buffer keys are **skipped during checkpoint load** so the target dataset's prompts are preserved — this is intentional for cross-dataset transfer.

### Bundled LLM weights

`model/chatglm-6b/` (8 shards) and `model/deepseek-r1-distill-llama-8B/` (2 shards) contain the full pretrained LLM weights (~360 GB total). These are loaded by the `Model` constructor and kept frozen.

### Key CLI flags

| Flag | Purpose |
|---|---|
| `--model` | Model name string (TimeLLM, BNT, BrainNetCNN, ALTER, GCDGCN) |
| `--dataset` | Dataset class name (e.g. `DiseaseCAUEEGDataset`) |
| `--llm_type` | `chatglm` or `llama` |
| `--llm_path` | Path to bundled LLM weights |
| `--pretrain_path` | Path to pretrained checkpoint for transfer learning |
| `--few_shot` | K subjects per class (0 = zero-shot) |
| `--use_dataset_prompt` / `--use_task_prompt` / `--use_stats_prompt` | Toggle prompt components |
| `--fc_threshold` / `--fc_keep_ratio` | FC matrix sparsification |
| `--deepspeed` / `--do_parallel` | Distributed backend selection |
