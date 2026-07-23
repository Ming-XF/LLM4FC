#!/bin/bash
# export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=3


# cd ../..
# python main.py \
# \
# --model "BrainNetCNN" \
# --within_subject \
# --num_repeat 5 \
# --subject_num 9 \
# \
# --dataset 'C42B' \
# --data_dir "../data/C42B/C42B128.npy" \
# --batch_size 32 \
# --num_epochs 200 \
# --drop_last True \
# --schedule 'cos' \
# --learning_rate 1e-3 \
# \
# --do_train \
# --do_evaluate \
# --do_test

python main.py --model "SrCVIB" --num_repeat 3 --dataset 'C42B' --data_dir "../data/C42B/C42B128.npy" --percentage 1. --batch_size 8 --num_epochs 300 --drop_last False --integration "add" --cor_comput "pearson" --d_model 64 --window_size 50 --window_stride 3 --dynamic_length 440 --abla_channel -1 --abla_vae "n" --num_layers 1 --schedule 'cos' --learning_rate 1e-5 --do_train --do_evaluate --do_test
