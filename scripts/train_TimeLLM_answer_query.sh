#!/bin/bash
#
# TimeLLM training script — experiment 1: pretrained LLM + answer query
#
# Replaces the flatten-all-patches readout with a trailing <answer> token whose
# hidden state alone feeds the prediction head. Per codex_output.txt, the main
# setting for any "LLM pretraining helps generalization" experiment must be
# --llm_init pretrained (keep train_TimeLLM.sh's --llm_init random as the
# random-init control).
#
# Usage: ./scripts/train_TimeLLM_answer_query.sh   (dataset / early-stop 参数在脚本内部配置)
#
# Datasets:  DiseaseBeirut / GenderDS  (AUC, 0.001)
#            AgeCAUEEG / AgeTUAB      (MAE, 0.1)
#

# ── 实验参数（在脚本内部配置，不再从命令行传入）──
DATASET="AgeCAUEEG"              # 数据集: DiseaseBeirut/GenderDS/AgeCAUEEG/AgeTUAB 等
EARLY_STOP_METRIC="MAE"          # 早停指标: AUC(分类) / MAE(回归)
PRETRAIN_PATH=""                 # 零样本评估: "./output_dir/TimeLLM_AgeCAUEEG_train"

deepspeed --num_gpus=2 main.py \
    --model "TimeLLM" \
    --num_epochs 100 \
    --num_repeat 1 \
    --save_steps 25 \
    --schedule 'cos' \
    --early_stop_patience 20 \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --dataset "$DATASET" \
    --fc_threshold 0.0 \
    --fc_keep_ratio 1.0 \
    --graph_evidence \
    --no_patches \
    --stats_head \
    --d_model 128 \
    --num_heads 8 \
    --d_ff 128 \
    --dropout 0.1 \
    --llm_type llama \
    --llm_path ./model/deepseek-r1-distill-llama-8B \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "$PRETRAIN_PATH" \
    --llm_init pretrained \
    --use_task_prompt \
    --readout evidence_query \
    --num_gcn_layers 1 \
    --num_prototypes 1000 \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
