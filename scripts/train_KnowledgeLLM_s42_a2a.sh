#!/bin/bash
#
# A2a ablation: random-initialized LLM backbone + fixed pretrained bank
# (pretrained_bank.npz).  Eval must rebuild the same random backbone, so the
# matching --seed is required in the eval script.
# Run: sh scripts/train_KnowledgeLLM_s42_a2a.sh
#

DATASET="DiseaseCAUEEG"
MODEL_DIR="output_dir/output_dir_knowledge_s42_a2a"
APPEND="_s42_caueeg_a2a"
KNOWLEDGE_BANK="output_dir/knowledge_banks/pretrained_bank.npz"
EARLY_STOP_METRIC="AUC"

cd /root/autodl-tmp/LLM4FC || exit 1
export LLM4FC_DATA_10S=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

deepspeed --master_port 29534 --num_gpus=2 main.py \
    --model "KnowledgeLLM" \
    --seed 42 \
    --num_epochs 200 \
    --num_repeat 1 \
    --save_steps 25 \
    --schedule 'cos' \
    --warmup_steps 400 \
    --early_stop_patience 20 \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --dataset "$DATASET" \
    --model_dir "$MODEL_DIR" \
    --append "$APPEND" \
    --few_shot 0 \
    --d_model 128 \
    --num_heads 8 \
    --dropout 0.1 \
    --llm_type llama \
    --llm_path ./model/deepseek-r1-distill-llama-8B \
    --llm_init random \
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
    --contrast_lambda 0.3 \
    --knowledge_contrast_temperature 0.2 \
    --deepspeed \
    --deepspeed_config deepspeed/train_patch.json \
    --do_train \
    --do_evaluate \
    --do_test
