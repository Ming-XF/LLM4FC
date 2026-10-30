#!/bin/bash
#
# TimeLLMKnowledge 10s zero-shot evaluation script (single-GPU, no training)
# Edit TARGET_DATASET / PRETRAIN_PATH / append before each run.
# Run: sh scripts/eval_TimeLLMKnowledge_10s_zs.sh
#

cd /root/autodl-tmp/LLM4FC || exit 1
export LLM4FC_DATA_10S=1
export CUDA_VISIBLE_DEVICES=1

TARGET_DATASET="DiseaseTUEP"
PRETRAIN_PATH="./output_dir_tllm10s_s42_tuab/TimeLLMKnowledge_DiseaseTUAB_train"
APPEND="_s42_tuab"

python main.py \
    --model "TimeLLMKnowledge" \
    --dataset "$TARGET_DATASET" \
    --pretrain_path "$PRETRAIN_PATH" \
    --append "$APPEND" \
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
    --contrast_lambda 0.1 \
    --freeze_head_bias \
    --use_semantic_prototypes \
    --knowledge_anchor_candidates 1000 \
    --knowledge_anchors 10 \
    --knowledge_description_candidates 1000 \
    --knowledge_description_prototypes 100 \
    --knowledge_similarity centered_cosine \
    --knowledge_aspect_mix 0.6 \
    --knowledge_anchor_fusion 0.1 \
    --use_patch_embeds \
    --do_test
