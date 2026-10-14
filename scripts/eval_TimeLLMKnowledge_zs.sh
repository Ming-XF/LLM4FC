#!/bin/bash
#
# TimeLLMKnowledge zero-shot evaluation script (single-GPU, no training)
# No external arguments; edit the variables below.
# Run: sh scripts/eval_TimeLLMKnowledge_zs.sh
#

TARGET_DATASET="DiseaseTUAB"
PRETRAIN_PATH="./output_dir/TimeLLMKnowledge_DiseaseCAUEEG_train"

python main.py \
    --model "TimeLLMKnowledge" \
    --dataset "$TARGET_DATASET" \
    --pretrain_path "$PRETRAIN_PATH" \
    --few_shot 0 \
    --batch_size 16 \
    --d_model 128 \
    --num_heads 8 \
    --dropout 0.1 \
    --llm_type llama \
    --llm_path ./model/deepseek-r1-distill-llama-8B \
    --llm_init pretrained \
    --use_task_prompt \
    --num_prototypes 1000 \
    --readout_layers 4 \
    --use_local_knowledge \
    --use_knowledge_bypass \
    --freeze_head_bias \
    --use_semantic_prototypes \
    --proto_preselect 1000 \
    --proto_per_phrase 100 \
    --use_patch_embeds \
    --do_test
