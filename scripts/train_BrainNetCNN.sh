#!/bin/bash
#
# BrainNetCNN training script
# Usage: ./scripts/train_BrainNetCNN.sh <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>
#
# Examples:
#   ./scripts/train_BrainNetCNN.sh DiseaseBeirut   ../data/Beirut/beirut_disease.npy     AUC
#   ./scripts/train_BrainNetCNN.sh GenderDS        ../data/DS/ds_gender.npz              AUC
#   ./scripts/train_BrainNetCNN.sh AgeTUAB         ../data/TUAB/tuab_age.npz             Loss
#   ./scripts/train_BrainNetCNN.sh FutureFCTUEP    ../data/TUEP/tuep_futurefc.npz        Loss
#

DATASET=${1:?"Usage: $0 <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>"}
DATA_DIR=${2:?"Usage: $0 <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>"}
EARLY_STOP_METRIC=${3:?"Usage: $0 <DATASET> <DATA_DIR> <EARLY_STOP_METRIC>"}

deepspeed --num_gpus=6 main.py \
    --model "BrainNetCNN" \
    --num_repeat 1 \
    --dataset "$DATASET" \
    --few_shot 0 \
    --few_shot_seed 42 \
    --pretrain_path "" \
    --data_dir "$DATA_DIR" \
    --fc_threshold 0.0 \
    --fc_keep_ratio 1.0 \
    --batch_size 2 \
    --num_epochs 200 \
    --drop_last False \
    --schedule 'cos' \
    --early_stop_patience 10 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "$EARLY_STOP_METRIC" \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
