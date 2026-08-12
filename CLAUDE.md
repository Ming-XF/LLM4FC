# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/claude) when working with code in this repository.

## Project Overview

LLM4FC is a research codebase for EEG-based brain disorder classification using LLM-augmented functional connectivity (FC) analysis. The core idea: compute dynamic FC matrices from EEG time series via sliding-window Pearson correlation → encode each window with a shared GCN → use a cross-attention **reprogramming layer** to map GCN outputs into a frozen LLM's embedding space → feed prompt embeddings + reprogrammed patches through the frozen LLM → task-specific head (classification / regression / future-FC prediction).

**Research goals (two tracks):**
1. **In-domain SOTA**: Achieve state-of-the-art on FC-based classification. TimeLLM already surpasses ALTER on CAUEEG disease diagnosis.
2. **Cross-domain generalization**: Make the model generalize across datasets/domains. Currently only FutureFC auxiliary loss helps here; planned additions include mask-token loss, contrastive loss, and two architectural proposals (see "Domain Generalization" section below).

## Commands

### Training

All training uses DeepSpeed ZeRO-2 with 6 GPUs via shell scripts:

```bash
# TimeLLM (primary model) — recommended way to train
./scripts/train_TimeLLM.sh <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>

# Examples:
./scripts/train_TimeLLM.sh DiseaseCAUEEG ../data/CAUEEG/caueeg_disease.npz AUC
./scripts/train_TimeLLM.sh DiseaseBeirut  ../data/Beirut/beirut_disease.npy  AUC
./scripts/train_TimeLLM.sh AgeTUAB        ../data/TUAB/tuab_age.npz          Loss
./scripts/train_TimeLLM.sh FutureFCTUEP   ../data/TUEP/tuep_futurefc.npz     Loss

# Legacy models (BNT, BrainNetCNN, GCDGCN, ALTER)
./scripts/train_ALTER.sh <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>
./scripts/train_BNT.sh <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>
```

### Testing / Evaluation

```bash
# Zero-shot evaluation with pretrained model
python main.py --model TimeLLM --dataset DiseaseCAUEEG \
  --pretrain_path output_dir/TimeLLM_DiseaseCAUEEG_train \
  --few_shot 0 --do_test

# Evaluate a trained model
python main.py --model TimeLLM --dataset DiseaseCAUEEG \
  --do_test --model_dir output_dir
```

### Manual Training Invocation (without script wrapper)

```bash
deepspeed --num_gpus=6 main.py \
  --model TimeLLM --dataset DiseaseCAUEEG --data_dir <path> \
  --llm_type llama --llm_path ./model/deepseek-r1-distill-llama-8B \
  --batch_size 2 --num_epochs 200 --deepspeed --do_train --do_evaluate --do_test \
  --use_dataset_prompt --use_task_prompt --block_causal_mask
```

### Few-shot Transfer Learning

```bash
# Full-data finetune (--few_shot -1) from pretrained checkpoint
deepspeed --num_gpus=6 main.py --model TimeLLM \
  --dataset DiseaseCAUEEG --data_dir <path> \
  --pretrain_path output_dir/TimeLLM_DiseaseCAUEEG_train \
  --few_shot -1 --few_shot_seed 42 --deepspeed --do_train

# N-shot per class (--few_shot N)
# --few_shot 0 = zero-shot (no training, just evaluate)
```

### Data Preprocessing

Preprocessing scripts convert raw EDF files to .npy/.npz format in `data/<dataset>/`:

```bash
python -m data.caueeg.disease_caueeg
python -m data.tuab.age_tuab
```

Use the `find-preprocess-params` skill to optimize `max_windows_per_subject` and `max_subjects` caps for new datasets.

## Architecture

```
main.py              — Entry: training loop, transfer learning, testing
config.py            — CLI args + model-config factory (init_model_config)
trainers.py          — Per-model Trainer subclasses (TimeLLMTrainer, etc.)
  └─ utils/trainer.py — Base Trainer: train/finetune/eval loops, early stopping,
                        metrics (binary/multi-class/regression/multi-output),
                        checkpoint save/load (DeepSpeed-aware)
utils/               — early_stopping, optimizer, schedule, recorder, logger
model/base/          — BaseConfig (hyperparams), ModelOutputs (logits + loss + hidden_state)
model/TimeLLM/       — TimeLLM model, prompts.py, gc_lora.py
model/<other>/       — ALTER, BNT, BrainNetCNN, GCDGCN (legacy/supplementary models)
data/                — BaseDataset + per-dataset implementations
  data/dataset.py    — DFC (sliding-window Pearson), FC sparsification, splits, few-shot
  data/preprocess.py — EA alignment, data normalization
scripts/             — Shell launch scripts (all use deepspeed --num_gpus=6)
deepspeed/           — train.json (ZeRO-2), finetune.json (lower LR for transfer)
```

### TimeLLM Forward Pipeline (detailed)

1. **Dynamic FC**: `BaseDataset.__getitem__` computes DFC via `dynamic_connectivity()` — sliding-window Pearson → (T, C, C), sparsified by `--fc_threshold` / `--fc_keep_ratio`
2. **GCN encoding**: DFC (B, T, C, C) flattened to (B*T, C, C) → shared GCN per window. Node init: one-hot channel identity via `channel_embed_projection` → (C, gcn_hidden), expanded to (B*T, C, gcn_hidden). Output: (B, T*C, gcn_hidden) in **time-first** order: C0T0, C1T0, ..., C18T0, C0T1, ..., C18T9
3. **Node projection**: (B, T*C, d_model) via Linear + LayerNorm + GELU + Dropout
4. **Reprogramming**: Cross-attention — node embeddings (Q), learned text prototypes (K,V). Maps d_model → 4096 (LLM hidden dim). Prototypes: 1000 learnable 4096-dim vectors projected from frozen LLM word embeddings via `mapping_layer`
5. **LLM forward**: Prompt assembly — `[start_tag | dataset_prompt? | task_prompt? | stats_prompt? | end_tag | reprogrammed_patches]`. Prompt parts toggled by `--use_dataset_prompt` / `--use_task_prompt` / `--use_stats_prompt`. Default attention is causal (future windows can't see past); `--block_causal_mask` makes same-window tokens bidirectional
6. **Task head** (patch tokens only, after P_skip):
   - Classification: Flatten → Linear(output_dim), CE loss
   - Regression: Flatten → Linear(1), MSE loss
   - Multi-output regression (FutureFC): Select future-window tokens → differentiable Pearson correlation head → (B, T_out, C, C), MSE loss
7. **FutureFC auxiliary loss** (`--futurefc_aux_weight > 0`): Predicts second-half DFC windows from LLM outputs, MSE × weight added to main loss. Works for classification tasks too.

### Token Budget

LLM input = P_start + P_dataset + P_task + P_stats + P_end + (T × C) = ~30–80 prompt + 190 patch tokens (10 windows × 19 channels) ≈ 220–300 tokens total.

### Dual LLM Backend

- **ChatGLM** (`--llm_type chatglm`): 2D position encoding, custom causal mask (True=masked), output (S, B, H) → `.transpose(0,1)`. Module paths: `transformer.layers[i].attention.<name>` / `transformer.layers[i].mlp.<name>`
- **LLaMA** (`--llm_type llama`): RoPE 1D position encoding, output (B, S, H) directly. Default model: DeepSeek-R1-Distill-Llama-8B. Module paths: `model.layers[i].self_attn.<name>` / `model.layers[i].mlp.<name>`

### GC-LoRA (Graph-Conditioned LoRA)

Inserts a one-hop GCN between LoRA matrices A and B in the low-rank bottleneck:

```
h = A(x)              # d_in → r
h_agg = FC_adj · h    # intra-window graph convolution along channel dim
Δ = B(h_agg)          # r → d_out
```

- `fc_adj` (B, T, C, C) is set via `set_gc_lora_context()` before LLM forward, cleared after
- When `use_graph_cond=False`, falls back to standard LoRA (shared code path for ablation)
- Enable: `--use_lora --use_gc_lora`; GC-LoRA requires LoRA
- Target modules: `--lora_target_modules "q_proj,v_proj"` (default)

### Key Architectural Decisions

- LLM is **frozen** (no fine-tuning). Trainable: GCN, node projection, reprogramming, mapping layer, output head, and optional LoRA/GC-LoRA weights
- `load_model` skips `dataset_prompt_embeddings` / `task_prompt_embeddings` buffers (preserves target-dataset prompts during transfer)
- When `--num_repeat >= 2`, uses GroupKFold; `--num_repeat == 1` uses fixed 60/20/20 per-subject stratified split (standard mode)
- Per-subject splitting prevents data leakage (all windows of a subject go to the same split)
- `few_shot_seed` controls which subjects are sampled during few-shot; different seeds → different subjects
- DeepSpeed config auto-detection: `deepspeed/train.json` (train), `deepspeed/finetune.json` (transfer; uses lower LR from JSON)

### Data Pipeline

Three task types controlled by `data_config.task_type`:
- `classification` — binary/multi-class disease or gender labels
- `regression` — scalar age prediction
- `multi_output_regression` — FutureFC prediction (matrix output, MSE loss, PearsonR metric)

`BaseDataset` supports: GroupKFold or fixed train/val/test splits, few-shot subject sampling, FC sparsification (threshold + top-K), DFC via `dynamic_connectivity()`, EA alignment via `preprocess_ea()`.

## Domain Generalization (Active Research Direction)

The current model achieves good in-domain performance but struggles with cross-dataset generalization. Two proposals under consideration:

### Proposal A: Domain-Adversarial Training (lower cost, implement first)
- Insert GRL + lightweight domain classifier after GCN output
- Forces GCN to learn domain-invariant features
- Risk: GRL λ scheduling is sensitive; domain labels may confound with task labels (e.g., Beirut = mostly older patients)

### Proposal B: Domain-Invariant Reprogramming (higher potential, higher cost)
- **Rationale**: Reprogramming is the sole "translator" into LLM space — if the same token position means different things across datasets, LLM knowledge is wasted
- **Approaches**:
  1. Prototype sharing + domain-specific offsets (lightweight domain encoder generates per-dataset perturbations)
  2. Contrastive alignment in reprogrammed token space (NT-Xent loss across datasets for same-class samples)
  3. Token-level domain classifier + GRL (deeper than Proposal A)
- **Challenge**: Requires multi-dataset mixed sampling in dataloader (currently single-dataset only)

### Recommended Strategy
1. Implement Proposal A first (fast validation on FC domain invariance)
2. Modify dataloader for multi-dataset training (prerequisite for Proposal B)
3. Explore Proposal B's contrastive/reprogram improvements as follow-up

### Evaluation Protocol
Use **leave-one-dataset-out**: train on N-1 datasets (with domain-adversarial/contrastive loss), zero-shot evaluate on the Nth.

## Adding a New Dataset

1. Create `data/<name>/` with a dataset class inheriting `BaseDataset`
2. Implement `load_data()` — loads .npy/.npz, sets `data_config.node_size`, `output_dim`, `task_type`, populates `all_data` with time_series / labels / subject_id
3. Implement `__getitem__` — returns `{'correlation': static_FC, 'DFC': dynamic_FC, 'labels': label}`
4. Register in `data/__init__.py`
5. Add prompt config in `model/TimeLLM/prompts.py` if using TimeLLM (channel names, groups, dataset-specific description)
6. Create a preprocessing script following the pattern in `data/caueeg/disease_caueeg.py`

## Available Datasets

| Dataset | EEG System | Tasks |
|---------|-----------|-------|
| CAUEEG | 19-ch 10-20 | Disease (AD/MCI/SCD/NC), Age |
| DS | 19-ch 10-20 | Disease (AD/HC), Age, Gender |
| TUAB | 19-ch 10-20 | Disease (abnormal/normal), Age, Gender |
| TUEP | 19-ch 10-20 | Disease (epilepsy/non), Age, Gender, FutureFC |
| Beirut | 19-ch 10-20 | Disease (epilepsy seizure prediction) |

## Important Implementation Notes

- Token order is **time-first**: C0T0, C1T0, ..., C18T0, C0T1, ..., C18T9. Don't change this without updating GC-LoRA reshape logic and block-causal mask
- The LLM forward uses `input_ids=None, inputs_embeds=...` (no text tokens, all embeddings)
- `ModelOutputs` has `.logits`, `.loss`, `.hidden_state` (dict with gcn_out, reprogrammed, HL_patches)
- DeepSpeed ZeRO-2 with `find_unused_parameters=True` (required by legacy BNT model)
- `Save_model`/`load_model` are DeepSpeed-aware (use `GatheredParameters` for rank-0 gather)
- GC-LoRA's `fc_adj` context is set/cleared per forward; leaking context between calls would cause silent bugs
- `num_repeat` doubles as K-Fold count; set to `1` for the standard train/val/test split
