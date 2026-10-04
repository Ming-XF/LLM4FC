#!/bin/bash
#
# TimeLLMPatch zero-shot evaluation script (single-GPU, no training)
# No external arguments; edit the variables below.
# Run: sh scripts/eval_TimeLLMPatch_zs.sh
#

TARGET_DATASET="AgeTUAB"
PRETRAIN_PATH="./output_dir/TimeLLMPatch_AgeCAUEEG_train"

python main.py \
    --model "TimeLLMPatch" \
    --dataset "$TARGET_DATASET" \
    --pretrain_path "$PRETRAIN_PATH" \
    --few_shot 0 \
    --batch_size 16 \
    --d_model 128 \
    --num_heads 8 \
    --d_ff 128 \
    --dropout 0.1 \
    --llm_type llama \
    --llm_path ./model/deepseek-r1-distill-llama-8B \
    --llm_init pretrained \
    --use_task_prompt \
    --num_prototypes 1000 \
    --patch_len 512 \
    --patch_stride 512 \
    --do_test
