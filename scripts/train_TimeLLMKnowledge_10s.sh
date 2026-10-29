#!/bin/bash
#
# TimeLLMKnowledge 10s-segment training script (win5 architecture,
# 7x4s/1s sliding knowledge windows; PatchStem temporal kernel 10).
# Edit DATASET between the three sequential runs (DiseaseCAUEEG/TUAB/TUEP).
# Run: sh scripts/train_TimeLLMKnowledge_10s.sh
#

DATASET="DiseaseTUEP"
EARLY_STOP_METRIC="AUC"

export LLM4FC_DATA_10S=1
export NCCL_P2P_DISABLE=1
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
    --model_dir output_dir_tllm10s \
    --append "_10s" \
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
    --deepspeed \
    --deepspeed_config deepspeed/train_patch.json \
    --do_train \
    --do_evaluate \
    --do_test
