#!/bin/bash

#存在问题：batchsize设置为16显存不够
#创新点：针对EEG信号的低信噪比问题，基于特征压缩的VAE方法；基于多视图的FC图构建方法
# python main.py --model "VAESTA" --num_repeat 3 --dataset 'Dementia400' --data_dir "../data/Dementia400/Dementia400.npy" --percentage 1. --batch_size 16 --num_epochs 200 --drop_last False --integration "add" --cor_comput "pearson" --d_model 64 --window_size 50 --window_stride 3 --dynamic_length 440 --num_heads 1 --num_layers 2 --learning_rate 0.0005 --max_learning_rate 0.001 --schedule 'one_cycle' --do_train --do_evaluate --do_test

# python main.py --model "VAESTA" --num_repeat 3 --dataset 'Dementia' --data_dir "../data/Dementia200/Dementia200.npy" --percentage 1. --batch_size 8 --num_epochs 200 --drop_last False --integration "add" --cor_comput "attention" --d_model 64 --window_size 50 --window_stride 3 --dynamic_length 440 --num_heads 1 --num_layers 2 --learning_rate 0.0005 --max_learning_rate 0.001 --schedule 'one_cycle' --do_train --do_evaluate --do_test




