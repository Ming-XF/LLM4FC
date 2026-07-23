#!/bin/bash

python main.py \
    --model "LDDE" \
    --num_repeat 1 \
    --dataset 'Dementia_MMS' \
    --data_dir "../data/XWDementia/Dementia_MMS.npy" \
    --batch_size 4 \
    --num_epochs 200 \
    --drop_last False \
    --train_set 0.8 \
    --val_set 0.1 \
    --schedule 'cos' \
    --learning_rate 1e-3 \
    --eps 1e-8 \
    --early_stop_patience 20 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Accuracy" \
    --do_train \
    --do_evaluate \
    --do_test
