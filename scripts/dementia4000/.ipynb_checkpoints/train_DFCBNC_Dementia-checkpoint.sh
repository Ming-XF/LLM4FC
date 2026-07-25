#!/bin/bash

python main.py \
    --model "DFCBNC" \
    --num_repeat 1 \
    --dataset 'Dementia4000' \
    --data_dir "../data/Dementia4000/Dementia4000.npz" \
    --batch_size 16 \
    --num_epochs 200 \
    --drop_last False \
    --train_set 0.6 \
    --val_set 0.2 \
    --schedule 'cos' \
    --learning_rate 1e-3 \
    --early_stop_patience 20 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Loss" \
    --do_train \
    --do_evaluate \
    --do_test
