#!/bin/bash

python main.py --model "ShallowConvNet" --num_repeat 3 --dataset 'Dementia' --data_dir "../data/Dementia200/Dementia200.npy" --batch_size 16 --num_epochs 200 --num_kernels 40 --drop_last False --model_dir "output_dir" --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test


