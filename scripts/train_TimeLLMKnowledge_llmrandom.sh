#!/bin/bash
#
# TimeLLMKnowledge llm_init=random 消融 (疾病任务)
# 与 train_TimeLLMKnowledge.sh 唯一差别: --llm_init random,
# --model_dir output_dir_llmrandom (隔离保存, 不覆盖 pretrained checkpoint),
# --append _llmrandom (区分日志)
# Run: sh scripts/train_TimeLLMKnowledge_llmrandom.sh
#

DATASET="DiseaseCAUEEG"
EARLY_STOP_METRIC="AUC"
MODEL_DIR="output_dir_llmrandom"

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
    --model_dir "$MODEL_DIR" \
    --append "_llmrandom" \
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
    --use_knowledge_bypass \
    --freeze_head_bias \
    --no-learn_layer_weights \
    --use_semantic_prototypes \
    --proto_preselect 1000 \
    --proto_per_phrase 100 \
    --no-use_proto_mixed_bank \
    --proto_mixed_gate 0.1 \
    --use_patch_embeds \
    --deepspeed \
    --deepspeed_config deepspeed/train_patch.json \
    --do_train \
    --do_evaluate \
    --do_test
