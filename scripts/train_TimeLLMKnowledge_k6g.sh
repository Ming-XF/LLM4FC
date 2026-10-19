#!/bin/bash
#
# TimeLLMKnowledge k6g 实验 training script
# k6 前端 + 融合前移(3×7 个 256 维组特征) + 读出头 256 维
# No external arguments; edit the variables below.
# Run: sh scripts/train_TimeLLMKnowledge_k6g.sh
#

DATASET="DiseaseCAUEEG"
EARLY_STOP_METRIC="AUC"

# H800 本机卡间 P2P 直连损坏（NCCL BROADCAST 无限挂起），必须禁用 P2P 走共享内存
export NCCL_P2P_DISABLE=1

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
    --model_dir output_dir/k6g \
    --append _k6g \
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
    --use_knowledge_fusion \
    --freeze_head_bias \
    --use_semantic_prototypes \
    --proto_preselect 1000 \
    --proto_per_phrase 100 \
    --use_patch_embeds \
    --deepspeed \
    --deepspeed_config deepspeed/train_patch.json \
    --do_train \
    --do_evaluate \
    --do_test
