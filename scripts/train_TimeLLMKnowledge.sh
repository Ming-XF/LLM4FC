#!/bin/bash
#
# TimeLLMKnowledge (领域知识 Token 编译器) training script
# No external arguments; edit the variables below.
# Run: sh scripts/train_TimeLLMKnowledge.sh
#

DATASET="DiseaseCAUEEG"
EARLY_STOP_METRIC="AUC"

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
    --model_dir output_dir/learn_layer_weights \
    --append "_learn_layer_weights" \
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
    --learn_layer_weights \
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
