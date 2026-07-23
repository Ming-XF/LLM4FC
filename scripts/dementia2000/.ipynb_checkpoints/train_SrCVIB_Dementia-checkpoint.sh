#!/bin/bash


python main.py --model "SrCVIB" --num_repeat 3 --dataset 'Dementia2000' --data_dir "../data/Dementia2000/Dementia2000.npy" --percentage 1. --batch_size 8 --num_epochs 1000 --drop_last False --integration "add" --cor_comput "pearson" --d_model 64 --window_size 50 --window_stride 3 --dynamic_length 440 --abla_channel -1 --abla_vae "n" --num_layers 1 --schedule 'cos' --learning_rate 1e-5 --do_train --do_evaluate --do_test




