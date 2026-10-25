#!/bin/bash
#
# TimeLLMKnowledge (领域知识 Token 编译器) training script
# No external arguments; edit the variables below.
# Run: sh scripts/train_TimeLLMKnowledge.sh
#

DATASET="DiseaseCAUEEG"
EARLY_STOP_METRIC="AUC"

# H800 本机卡间 P2P 直连损坏（NCCL BROADCAST 无限挂起），必须禁用 P2P 走共享内存
export NCCL_P2P_DISABLE=1
# 避免 libgomp 线程警告
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

deepspeed --num_gpus=2 main.py \
    --model "TimeLLMKnowledge" \
    --num_epochs 200 \
    --num_repeat 1 \
    --save_steps 25 \
    --schedule 'cos' \
    --warmup_steps 400 \
    --early_stop_patience 20 \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --dataset "$DATASET" \
    --append "_apimean300" \
    --model_dir output_dir/apimean300 \
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
    --anchor_select api \
    --gen_num_keywords 300 \
    --knowledge_anchor_candidates 1000 \
    --knowledge_anchors 10 \
    --knowledge_description_candidates 1000 \
    --knowledge_description_prototypes 100 \
    --knowledge_similarity centered_cosine \
    --knowledge_aspect_mix 0.6 \
    --knowledge_anchor_fusion 0.1 \
    --use_patch_embeds \
    --deepspeed \
    --deepspeed_config deepspeed/train_patch.json \
    --do_train \
    --do_evaluate \
    --do_test
