#!/bin/bash
#
# TimeLLMKnowledge llm_init=random 消融 — 零样本评估 (单卡, 无训练)
# 与 eval_TimeLLMKnowledge_zs.sh 唯一差别: pretrain 路径指向 llmrandom 保存目录,
# --llm_init random (与训练一致), --append _llmrandom
# Run: sh scripts/eval_TimeLLMKnowledge_llmrandom_zs.sh
#

TARGET_DATASET="DiseaseTUAB"
PRETRAIN_PATH="./output_dir_llmrandom/TimeLLMKnowledge_DiseaseCAUEEG_train"

python main.py \
    --model "TimeLLMKnowledge" \
    --dataset "$TARGET_DATASET" \
    --pretrain_path "$PRETRAIN_PATH" \
    --few_shot 0 \
    --append "_llmrandom" \
    --batch_size 16 \
    --d_model 128 \
    --num_heads 8 \
    --dropout 0.1 \
    --llm_type llama \
    --llm_path ./model/deepseek-r1-distill-llama-8B \
    --llm_init random \
    --use_task_prompt \
    --num_prototypes 1000 \
    --readout_layers 4 \
    --use_local_knowledge \
    --use_knowledge_fusion \
    --freeze_head_bias \
    --use_semantic_prototypes \
    --proto_preselect 1000 \
    --proto_per_phrase 100 \
    --use_patch_embeds \
    --do_test
