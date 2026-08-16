# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/claude) when working with code in this repository.

## Project Overview

LLM4FC is a research codebase for EEG-based brain-disorder analysis that routes **dynamic functional connectivity (DFC)** through a frozen LLM. Pipeline: EEG time series → sliding-window Pearson correlation (DFC, per window) + static FC (SFC) → a shared GCN encodes each window → a cross-attention **reprogramming layer** maps GCN outputs into the frozen LLM's token space → prompt + reprogrammed patches go through the LLM → a task head (classification / regression).

Two research goals:
1. **In-domain SOTA** — TimeLLM already beats ALTER on CAUEEG disease diagnosis.
2. **Cross-domain generalization** — currently only a FutureFC auxiliary loss helps; planned work (see `plan.txt`) includes mask-token loss, contrastive learning, and domain-adversarial training (a token-level GRL + domain classifier is already scaffolded in `model/TimeLLM/TimeLLM.py`, gated by `--use_token_domain_grl`).

## Commands

There is no test suite, linter config, or `requirements.txt`. Training is launched through `main.py` (single entry point) or the `scripts/*.sh` wrappers, all of which use DeepSpeed ZeRO-2 on 6 GPUs.

```bash
# Train (scripts take exactly 2 positional args: <DATASET> <EARLY_STOP_METRIC>)
./scripts/train_TimeLLM.sh DiseaseCAUEEG AUC
./scripts/train_TimeLLM.sh AgeTUAB Loss
./scripts/train_ALTER.sh DiseaseCAUEEG AUC   # also train_BNT.sh / train_BrainNetCNN.sh / train_GCDGCN.sh
```

Equivalent direct invocation (single GPU: drop `--deepspeed`; then AMP GradScaler is used):

```bash
deepspeed --num_gpus=6 main.py --model TimeLLM --dataset DiseaseCAUEEG \
  --llm_type llama --llm_path ./model/deepseek-r1-distill-llama-8B \
  --deepspeed --do_train --do_evaluate --do_test
```

Transfer learning (few-shot / zero-shot) is driven by `--pretrain_path` + `--few_shot`:
- `--few_shot 0` → zero-shot (load checkpoint, evaluate only)
- `--few_shot N` (>0) → sample N subjects/class, finetune, evaluate
- `--few_shot -1` → full-data finetune
- `--few_shot_seed` varies subject sampling across episodes; `main.py` runs one episode per invocation (collect results by re-running)

```bash
deepspeed --num_gpus=6 main.py --model TimeLLM --dataset DiseaseCAUEEG \
  --pretrain_path output_dir/TimeLLM_DiseaseCAUEEG_train --few_shot 20 --few_shot_seed 42 \
  --deepspeed --do_train
```

Data preprocessing: each dataset module has a `*_preprocess()` function and an `if __name__ == "__main__"` block that converts raw EDFs into a `.npz` cache (run with `python -m data.caueeg.disease_caueeg`, etc.).

## Architecture

**Data flow** (`main.py` → `trainers.py` → `utils/trainer.py` → `data/` → `model/`):

- `config.py` — two responsibilities: `init_config()` defines every CLI flag via argparse; `init_model_config(args, data_config)` builds a model + its config by name.
- `utils/trainer.py` — base `Trainer`. Constructs `DataConfig`, loads dataloaders, instantiates the model, and owns train / finetune / evaluate loops, early stopping, metric computation (binary / multi-class / regression), and DeepSpeed-aware checkpoint save/load. Dispatch: `main.py` calls `eval(args.model + 'Trainer')`.
- `trainers.py` — one thin `*Trainer` subclass per model, overriding only `prepare_inputs_kwargs` to reshape the raw batch into the model's expected keyword inputs.
- `model/base/` — `BaseConfig` (hyperparams), `BaseModel`, and `ModelOutputs(logits, loss, hidden_state)` — the return contract every model's `forward` must follow.
- `model/TimeLLM/` — the primary model (`TimeLLM.py`), plus `prompts.py` (per-dataset text prompts) and `gc_lora.py` (LoRA / GC-LoRA injection into the frozen LLM).
- `model/<BNT|BrainNetCNN|ALTER|GCDGCN>/` — supplementary/legacy models.
- `model/chatglm-6b/` and `model/deepseek-r1-distill-llama-8B/` — frozen LLM weights (gitignored).
- `data/dataset.py` — `BaseDataset`: split logic (GroupKFold when `num_repeat >= 2`, else 60/20/20 subject-level train/val/test), few-shot subject sampling, and on-the-fly SFC/DFC computation (nilearn Pearson) with FC sparsification (`--fc_threshold`, `--fc_keep_ratio`).
- `data/dataloader.py` — three loader factories: `init_StratifiedKFold_dataloader`, `init_distributed_dataloader`, `init_deepspeed_dataloader`.
- `data/<site>/` — per-site dataset classes (`disease_*`, `age_*`, `gender_*`); `data/multidomain/` — cross-dataset fusions that emit an extra `domain_label`.

## Conventions & Gotchas

- **Everything is registered by name string.** Models: `eval(args.model + 'Trainer')` (trainers.py) and a branch in `config.py::init_model_config`. Datasets: `eval(args.dataset + 'Dataset')` via imports in `data/__init__.py`. TimeLLM prompts: `prompts.py::DATASET_PROMPTS` keyed by dataset name (falls back to CAUEEG). Adding a model/dataset requires wiring all of these.
- **Data files** are `.npz` caches under `../data/<SITE>/` (note the hardcoded relative path — run from the repo root). Keys: `timeseries`, `labels`, `subject_id`, `hz`, and optionally precomputed `split_train_index` / `split_val_index` / `split_test_index`. Each dataset's `load_data` sets `data_config.output_dim` and `data_config.task_type` (`classification` vs `regression` for age prediction).
- **Batch dict** returned by `__getitem__`: `DFC` (W×N×N), `correlation` (SFC, kept for backward compat), `labels` (one-hot for classification), optionally `domain_label`.
- **Model output contract**: always return `ModelOutputs(logits=..., loss=..., hidden_state=...)`; `Trainer._forward` reads `.loss` and evaluators read `.logits`.
- **Frozen LLM**: the LLM backbone is loaded in bfloat16 and all its params are frozen; only the GCN, reprogramming, and head train. LoRA/GC-LoRA (`--use_lora`, `--use_gc_lora`) re-enables a small trainable subset. `--use_gc_lora` requires `--use_lora` (silently disabled otherwise).
- **DeepSpeed**: config auto-resolves to `deepspeed/train.json` (ZeRO-2); finetune swaps in `deepspeed/finetune.json` and overrides LR. The DeepSpeed launcher sets `RANK`/`WORLD_SIZE`, which `init_deepspeed_dataloader` reads before `deepspeed.initialize()` runs.
- **Early stopping**: `--early_stop_metric` (Accuracy / AUC / Loss / R2 / PearsonR / …) selects what to monitor; improvement direction is inferred from the metric name. `--early_stop_patience 0` disables it.
- **Two LLM backends** (`--llm_type chatglm|llama`) have different output layouts and mask conventions — ChatGLM returns `(S, B, H)` (needs transpose) and uses boolean masks; Llama returns `(B, S, H)` and uses float masks. Branch logic lives in `TimeLLM.py::forward`.
