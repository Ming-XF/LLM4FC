# LLM4FC Agent Guide

## Project Purpose

LLM4FC studies whether a model trained on EEG dynamic functional connectivity (DFC) can generalize across datasets, especially in zero-shot transfer. The main model is a TimeLLM-style pipeline; the repository also contains graph and temporal baselines.

Keep these research settings separate:

- Same-domain performance: train, validate, and test within one dataset.
- Cross-domain transfer: train on a source dataset and evaluate on a different target dataset with `--pretrain_path`.
- Zero-shot: transfer with `--few_shot 0`; no target-domain fine-tuning.
- Few-shot: transfer with a positive `--few_shot`; subjects, not individual windows, are sampled for fine-tuning.

Do not describe a good same-domain result as cross-dataset generalization.

## Repository Map

- `main.py`: runtime mode selection, seed setup, DeepSpeed configuration, train/test/transfer entry point.
- `config.py`: CLI arguments and model configuration factory.
- `trainers.py`: model-specific input adapters.
- `utils/trainer.py`: training, evaluation, early stopping, checkpoint save/load, and few-shot fine-tuning.
- `data/dataset.py`: subject-level splitting, few-shot sampling, static FC and DFC helpers.
- `data/<dataset>/`: dataset loaders and preprocessing for CAUEEG, DS, TUAB, TUEP, Beirut, and multi-domain datasets.
- `model/TimeLLM/TimeLLM.py`: TimeLLM model, GCN, reprogramming, prompts, CVIB, and optional temporal encoder.
- `model/TimeLLM/gc_lora.py`: standard LoRA and graph-conditioned GC-LoRA wrappers.
- `model/TimeLLM/prompts.py`: dataset/task prompt definitions.
- `scripts/`: training launch scripts.
- `log_dir/`: committed experiment logs, including historical experiments.
- `output_dir/`: checkpoints and saved configurations; files may come from older model versions.

## Data and DFC

The usual representation is a one-minute EEG segment with 19 channels sampled at 200 Hz. Each sample is converted to:

- SFC: one static Pearson-correlation matrix.
- DFC: ten sliding-window Pearson-correlation matrices, normally shaped `(T=10, C=19, C=19)`.

Dataset loaders generally use subject-grouped train/validation/test splits. This prevents windows from the same subject appearing in different splits. Few-shot selection is also subject-level.

Important caveats:

- TimeLLM receives DFC as its main input. SFC is returned by loaders but is not materially used by the current TimeLLM forward path.
- DFC is passed to the GCN as `adj_norm`, but the current implementation does not apply an additional standard graph normalization there.
- Dataset preprocessing is not identical. TUAB/TUEP use average reference while CAUEEG/DS use different preprocessing paths; this is a likely domain-shift source.
- Raw datasets are outside this repository, normally under `/root/autodl-tmp/data/`. Do not assume they are available in a clean checkout.

Age labels use fixed prior normalization bounds `[0, 100]` from `data/preprocess.py`, then regression metrics are converted back to years. Do not replace this with a data-fitted min/max when investigating transfer.

## Current TimeLLM Pipeline

The current forward path is:

```text
DFC (B,T,C,C)
  -> shared GCN for each time window
  -> 19 channel nodes x 10 temporal tokens
  -> optional per-node temporal Transformer
  -> mapping layer + reprogramming cross-attention
  -> prompt embeddings
  -> frozen ChatGLM/LLaMA backbone
  -> patch hidden states (first d_ff dimensions)
  -> flattened classification/regression head
```

The GCN starts from one-hot channel identity projected to `d_model`. Token order is controlled by `--token_order` (`time_first` by default, or `node_first`). The LLM is normally frozen. Cross-domain behavior is therefore primarily determined by the GCN, mapping/reprogramming layers, output head, and any enabled LoRA/CVIB/temporal components; language pretraining is not automatically EEG functional-connectivity pretraining.

Classification uses cross-entropy. Regression uses Huber loss with `delta=1.0` inside the model forward pass.

## Transfer and Checkpoints

In `main.py`:

- Empty `--pretrain_path`: ordinary training.
- `--pretrain_path PATH --few_shot 0`: zero-shot evaluation of the loaded source model on the target dataset.
- `--few_shot N` with `N > 0`: target-domain few-shot fine-tuning; validation is performed on the full target validation set and early stopping selects the checkpoint.
- `--few_shot -1`: full-data target fine-tuning.

Checkpoints under `output_dir/` save trainable parameters and buffers, not the frozen LLM weights. `load_model()` skips `dataset_prompt_embeddings` and `task_prompt_embeddings`, intentionally keeping the target dataset's prompt buffers during transfer. A checkpoint can therefore load only a subset of current model keys when source and target configurations or Git revisions differ.

When reading a checkpoint or log, always record:

1. source and target dataset;
2. zero-shot/few-shot/full-data mode and few-shot seed;
3. Git commit that produced the run;
4. `llm_init`, `use_lora`, `use_gc_lora`, target modules, injected layers, CVIB, temporal encoder, token order, and dimensions;
5. data-file/preprocessing generation version.

Do not infer a complete configuration from an output filename alone.

## Experiment Interpretation

The latest `scripts/train_TimeLLM.sh` contains `--llm_init random`. This is an explicit ablation, not the pretrained-LLM main setting. `config.py` still defaults to `llm_init="pretrained"`; use that setting when referring to the main pretrained experiment.

Several options in the latest script appear only in comments. Merely supplying LoRA hyperparameters does not enable LoRA: `--use_lora` is required, and `--use_gc_lora` additionally requires `--use_lora`. Likewise, `--temporal_heads` and `--temporal_layers` do not enable the temporal encoder without `--use_temporal_encoder`.

Relevant historical boundaries:

- `146989d`: early same-domain and cross-domain classification experiments; cross-domain classification was poor.
- `4fe8e4d`: standard LoRA/GC-LoRA classification experiments; no stable improvement.
- `ab1e63c`: fixed age normalization, Huber regression, and revised early stopping; age transfer logs should be interpreted from this version onward.
- `f78db1e`: AgeTUAB CVIB comparison.
- `48da400`: temporal encoder and `llm_init=random` ablations.

Observed pattern from committed logs: classification has useful same-domain signal but CAUEEG2<->DS transfer AUC is generally near random or worse, including one-shot fine-tuning. Age transfer degrades less and few-shot fine-tuning can help, but recorded zero-shot age results do not establish strong zero-shot generalization. Report this distinction explicitly.

## Development Workflow

1. Read the relevant current source and the Git commit that produced the target log/checkpoint before changing code.
2. Preserve unrelated user changes and do not rewrite historical logs or checkpoints.
3. Keep changes scoped to the requested model/data/training path.
4. Prefer existing repository conventions and helpers over new abstractions.
5. For dataset or transfer changes, inspect subject IDs, split indices, label normalization, prompt buffers, and checkpoint key loading together.
6. Avoid launching multi-GPU training or regenerating raw preprocessing unless explicitly requested; these jobs are expensive and raw data is external.

There is no maintained unit-test suite, dependency manifest, or general linter. At minimum, use syntax/import checks and a small CPU or mocked forward-path check for code changes. For training changes, inspect the generated log's configuration, final validation/test block, and checkpoint loading messages.

## Common Commands

```bash
# Inspect repository and experiment history
git status --short
git log --oneline --decorate -20

# Inspect current experiment logs
rg -n "Zero-shot|shot|test.*dataset|AUC:|Accuracy:|MAE:|RMSE:|R2:|PearsonR:" log_dir

# Main pretrained-LLM setting; adjust dataset and metric as needed
# The checked-in script currently uses random initialization as an ablation.
python main.py --model TimeLLM --dataset AgeCAUEEG \
  --llm_type llama --llm_path ./model/deepseek-r1-distill-llama-8B \
  --llm_init pretrained --few_shot 0 --do_train --do_evaluate --do_test

# Transfer mode conceptually
python main.py --model TimeLLM \
  --pretrain_path ./output_dir/TimeLLM_AgeCAUEEG_train \
  --dataset AgeTUAB --few_shot 0 --do_train
```

For real multi-GPU runs, use the appropriate `deepspeed` launcher and repository DeepSpeed JSON. `main.py` treats `deepspeed/train.json` and `deepspeed/finetune.json` as the source of truth for batch size and learning rate.

## Safety and Reproducibility

- Do not use destructive Git commands such as `git reset --hard` or `git checkout --` unless explicitly requested.
- Do not delete or overwrite `log_dir/` or `output_dir/` while investigating results.
- `main.py` sets seed 42 and deterministic CUDA-related options; few-shot episodes use `--few_shot_seed`.
- A single committed log is one run, not a variance estimate. Avoid claiming statistical significance without repeated seeds.
