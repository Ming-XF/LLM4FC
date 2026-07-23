#!/bin/bash

python main.py --model "LMDA" --num_repeat 3 --dataset 'Dementia400' --data_dir "../data/Dementia400/Dementia400.npy" --batch_size 16 --num_epochs 200 --num_kernels 24 --drop_last False --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test


