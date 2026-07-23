#!/bin/bash

python main.py --wandb_entity cwg --project Dementia --model "EEGNet" --num_repeat 3 --dataset 'Dementia' --data_dir "../data/Dementia200/Dementia200.npy" --batch_size 16 --num_epochs 200 --frequency 128 --D 22 --num_kernels 16 --p1 4 --p2 8 --dropout 0.5 --drop_last False --model_dir "output_dir" --schedule "cos" --learning_rate 1e-3 --do_train --do_evaluate --do_test


