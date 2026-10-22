#!/bin/bash
#
# TimeLLMKnowledge zero-shot evaluation script (single-GPU, no training)
# No external arguments; edit the variables below.
# Run: sh scripts/eval_TimeLLMKnowledge_zs.sh
#

TARGET_DATASET="DiseaseTUEP"
PRETRAIN_PATH="./output_dir/v3/TimeLLMKnowledge_DiseaseCAUEEG_train"

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
    --use_local_knowledge \
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
    --append _v3 \
    --do_test
