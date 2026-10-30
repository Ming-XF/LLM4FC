#!/bin/bash
#
# EEGNet 10s multi-seed zero-shot eval — runs ALL 6 cells for the seed.
# Edit SEED before each run.
# Run: sh scripts/eval_EEGNet_10s_mseed.sh
#

SEED=44

cd /root/autodl-tmp/LLM4FC || exit 1

export LLM4FC_DATA_10S=1
export CUDA_VISIBLE_DEVICES=0

for PAIR in "DiseaseCAUEEG:DiseaseTUAB" "DiseaseCAUEEG:DiseaseTUEP" \
            "DiseaseTUAB:DiseaseCAUEEG" "DiseaseTUAB:DiseaseTUEP" \
            "DiseaseTUEP:DiseaseCAUEEG" "DiseaseTUEP:DiseaseTUAB"; do
    SRC="${PAIR%%:*}"
    TGT="${PAIR##*:}"
    SRC_TAG=$(echo "$SRC" | sed 's/Disease//' | tr 'A-Z' 'a-z')
    python main.py \
        --model "EEGNet" \
        --dataset "$TGT" \
        --pretrain_path "./output_dir/output_dir_eegnet_s${SEED}/EEGNet_${SRC}_train" \
        --few_shot 0 \
        --batch_size 16 \
        --frequency 128 \
        --D 2 \
        --num_kernels 8 \
        --p1 4 \
        --p2 8 \
        --dropout 0.5 \
        --append "_s${SEED}_${SRC_TAG}" \
        --do_test
done
