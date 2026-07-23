#!/bin/bash

python main.py --model "CEEDNet" --num_repeat 3 --dataset 'Dementia4000' --data_dir "../data/Dementia4000/Dementia4000.npy" --batch_size 16 --num_epochs 200 --drop_last False --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test


