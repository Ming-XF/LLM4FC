#!/bin/bash
#
# A3a (main method: no_raw) zero-shot evaluation: CAUEEG (trained) -> TUAB.
# Must pass the same --knowledge_modalities as training.
# Run: sh scripts/eval_KnowledgeLLM_s42_a3a_caueeg_to_tuab.sh
#

TARGET_DATASET="DiseaseTUAB"
PRETRAIN_PATH="./output_dir/output_dir_knowledge_s42_a3a/KnowledgeLLM_DiseaseCAUEEG_train"
APPEND="_s42_a3a_caueeg_to_tuab"
KNOWLEDGE_BANK=""

cd /root/autodl-tmp/LLM4FC || exit 1
export LLM4FC_DATA_10S=1
export CUDA_VISIBLE_DEVICES=0

python main.py \
    --model "KnowledgeLLM" \
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
    --knowledge_bank "$KNOWLEDGE_BANK" \
    --knowledge_top_k 8 \
    --knowledge_top_m 32 \
    --knowledge_temperature 1.0 \
    --knowledge_fusion_tokens 16 \
    --knowledge_raw_patch 200 \
    --knowledge_raw_stride 200 \
    --knowledge_psd_patch 8 \
    --knowledge_psd_stride 4 \
    --knowledge_fc_patch 1 \
    --knowledge_fc_stride 1 \
    --knowledge_llm_layers 6 \
    --knowledge_modalities no_raw \
    --contrast_lambda 0.3 \
    --knowledge_contrast_temperature 0.2 \
    --do_test
