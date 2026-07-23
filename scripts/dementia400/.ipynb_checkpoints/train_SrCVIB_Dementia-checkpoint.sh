#!/bin/bash


#存在问题：batchsize设置为16显存不够
#创新点：针对EEG信号的低信噪比问题，基于特征压缩的VAE方法；基于多视图的FC图构建方法
python main.py --model "SrCVIB" --num_repeat 3 --dataset 'Dementia400' --data_dir "../data/Dementia400/Dementia400.npy" --percentage 1. --batch_size 8 --num_epochs 700 --drop_last False --integration "add" --cor_comput "pearson" --d_model 64 --window_size 50 --window_stride 3 --dynamic_length 440 --abla_channel -1 --abla_vae "n" --num_layers 1 --schedule 'cos' --learning_rate 1e-5 --do_train --do_evaluate --do_test




