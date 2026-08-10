# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM4FC is a research codebase for EEG-based brain disorder classification using LLM-augmented functional connectivity (FC) analysis. The core idea: compute dynamic FC matrices from EEG time series via sliding-window Pearson correlation → encode each window with a shared GCN → use a cross-attention **reprogramming layer** to map GCN outputs into a frozen LLM's embedding space → feed prompt embeddings + reprogrammed patches through the frozen LLM → task-specific head (classification/regression/future-FC prediction).

## Commands

Training (all via shell scripts with DeepSpeed ZeRO-2, 6 GPUs):

```bash
# TimeLLM (main model)
./scripts/train_TimeLLM.sh <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>
# Example: ./scripts/train_TimeLLM.sh DiseaseBeirut ../data/Beirut/beirut_disease.npy AUC

# Other models (BNT, BrainNetCNN, GCDGCN, ALTER)
./scripts/train_BNT.sh DiseaseBeirut ../data/Beirut/beirut_disease.npy AUC
```

Key arguments: `--model TimeLLM`, `--dataset`, `--llm_type chatglm|llama`, `--deepspeed`, `--do_train`, `--pretrain_path` (for transfer learning), `--few_shot N` (-1=full finetune, 0=zero-shot, >0=N-shot per class), `--futurefc_aux_weight` (auxiliary loss weight), `--fc_threshold`/`--fc_keep_ratio` (FC sparsification), `--use_dataset_prompt`/`--use_task_prompt`/`--use_stats_prompt` (prompt toggles).

The DeepSpeed config at `deepspeed/train.json` (ZeRO-2, batch size 12, 2 per GPU) is auto-detected; `deepspeed/finetune.json` is used for transfer-learning finetune.

## Architecture

```
main.py          — Entry point: training / transfer learning / testing
config.py        — CLI argument parser + model-config factory (init_model_config)
trainers.py      — Per-model Trainer subclasses (TimeLLMTrainer, BNTTrainer, etc.)
utils/trainer.py — Base Trainer: training loop, early stopping, eval (binary/multi/regression/multi-output), save/load
utils/           — early_stopping, optimizer, schedule, recorder, logger
model/base/      — BaseConfig (hyperparams), ModelOutputs dataclass (logits + loss + hidden_state)
model/TimeLLM/   — TimeLLM model + prompts.py (per-dataset prompt configs)
model/<other>/   — BNT, BrainNetCNN, ALTER, GCDGCN (legacy models)
data/            — BaseDataset + per-dataset implementations
data/dataset.py  — DFC computation (sliding-window Pearson), FC sparsification, per-subject data splits, few-shot sampling
data/preprocess.py — EEG preprocessing (EDF → .npy with time series, labels, subject IDs)
scripts/          — Shell launch scripts (all use deepspeed --num_gpus=6)
deepspeed/        — DeepSpeed ZeRO-2 configs (train.json, finetune.json)
```

### TimeLLM Forward Pipeline

1. **Dynamic FC**: `BaseDataset.__getitem__` computes DFC via `dynamic_connectivity()` — sliding-window Pearson → (T, C, C) tensor, sparsified by threshold/top-K
2. **GCN encoding**: DFC (B, T, C, C) → shared GCN per window → (B, T*C, gcn_hidden). Node init: one-hot channel identity via `channel_embed_projection`. Token order: **time-first** (C0T0, C1T0, ..., C18T0, C0T1, ..., C18T9)
3. **Node projection**: (B, T*C, gcn_hidden) → (B, T*C, d_model) via Linear+LayerNorm+GELU
4. **Reprogramming**: Cross-attention from node embeddings (Q) to learned text prototypes (K,V) — maps d_model→4096 (LLM dim)
5. **LLM forward**: `[start_tag | dataset_prompt? | task_prompt? | stats_prompt? | end_tag | reprogrammed_patches]` → frozen LLM. Prompt parts are optional (toggled by CLI flags). Uses causal attention so future windows cannot attend past ones.
6. **Task head**:
   - Classification/Regression: Flatten all patch tokens → Linear(output_dim)
   - Multi-output regression (FutureFC): Select future-window node tokens → differentiable Pearson correlation head → (B, T_out, C, C), MSE loss against ground-truth future FC
7. **FutureFC auxiliary loss** (for non-FutureFC tasks): predicts the second half of DFC windows from LLM outputs, MSE-weighted and added to main loss

### Data Pipeline

`BaseDataset` supports three task types via `data_config.task_type`:
- `classification` — binary or multi-class disease/gender labels
- `regression` — scalar age prediction
- `multi_output_regression` — FutureFC prediction (matrix output)

Data splitting: when `--num_repeat >= 2`, uses GroupKFold; when `--num_repeat == 1`, uses a fixed 60/20/20 per-subject stratified split (the standard mode). Few-shot sampling selects N subjects per class from the training set.

### Dual LLM Backend Support

The model supports two LLM backends with different position encoding and output shapes:

- **ChatGLM** (`--llm_type chatglm`): Uses 2D position encoding, custom causal mask (True=masked), output shape (S, B, H) requiring `.transpose(0,1)`. Module path: `transformer.layers[i].attention.<name>` / `transformer.layers[i].mlp.<name>`.
- **LLaMA** (`--llm_type llama`): Uses RoPE 1D position encoding, no explicit attention mask needed, output shape (B, S, H) directly. Module path: `model.layers[i].self_attn.<name>` / `model.layers[i].mlp.<name>`.

### GC-LoRA (Graph-Conditioned LoRA)

`model/TimeLLM/gc_lora.py` — Extends standard LoRA by inserting a one-hop GCN aggregation between the low-rank matrices A and B:

```
h = A(x)           # d_in → r
h_agg = FC_adj · h # intra-window graph convolution along channel dim
Δ = B(h_agg)       # r → d_out
```

Key details:
- The GCN sits in the low-rank bottleneck (dimension r), so FLOPs are negligible
- `fc_adj` (B, T, C, C) is set as runtime context via `set_gc_lora_context()` before each LLM forward and cleared via `clear_gc_lora_context()` after
- When `use_graph_cond=False`, falls back to standard LoRA (shares code path for ablation)
- Enabled with `--use_lora --use_gc_lora`; `--use_gc_lora` requires `--use_lora`
- Target modules specified via `--lora_target_modules` (default: `q_proj,v_proj`)

### Token Budget in LLM Forward

The LLM input sequence length = `P_start + P_dataset + P_task + P_stats + P_end + (T * C)`. With 10 windows × 19 channels = 190 patch tokens plus prompt overhead (~30–80 tokens), total is ~220–300 tokens.

### Key Design Choices

- LLM is **frozen** (no fine-tuning). Only the GCN, node projection, reprogramming layer, mapping layer, and output head are trained.
- `load_model` skips `dataset_prompt_embeddings` and `task_prompt_embeddings` buffers (keeps target-dataset prompts when transferring).
- DeepSpeed ZeRO-2 is the primary distributed training strategy.
- The `num_repeat` argument doubles as the K-Fold split count; set to 1 for train/val/test mode.
- `early_stop_metric` chooses the monitored metric; `<trainer>.train()` infers direction (min for Loss, max for Accuracy/AUC).

### Adding a New Dataset

1. Create `data/<name>/` with a dataset class inheriting `BaseDataset`
2. Implement `load_data()` — loads .npy/.npz, sets `data_config.node_size`, `output_dim`, `task_type`, and populates `all_data` with time_series/labels/subject_id
3. Implement `__getitem__` — returns `{'correlation': static_FC, 'DFC': dynamic_FC, 'labels': label}`
4. Register in `data/__init__.py`
5. Add prompt config to `model/TimeLLM/prompts.py` if using TimeLLM
