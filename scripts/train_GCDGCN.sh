#!/bin/bash
#
# GCDGCN training script
# Usage: ./scripts/train_GCDGCN.sh <DATASET> <DATA_DIR>
#
# Examples:
#   ./scripts/train_GCDGCN.sh DiseaseBeirut   ../data/Beirut/beirut_disease.npy
#   ./scripts/train_GCDGCN.sh GenderDS        ../data/DS/ds_gender.npz
#   ./scripts/train_GCDGCN.sh AgeTUAB         ../data/TUAB/tuab_age.npz
#   ./scripts/train_GCDGCN.sh FutureFCTUEP    ../data/TUEP/tuep_futurefc.npz
#

DATASET=${1:?"Usage: $0 <DATASET> <DATA_DIR>"}
DATA_DIR=${2:?"Usage: $0 <DATASET> <DATA_DIR>"}

deepspeed --num_gpus=6 main.py \
    --model "GCDGCN" \
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
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --early_stop_patience 25 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "AUC" \
    --deepspeed \
    --do_train \
    --do_evaluate \
    --do_test
