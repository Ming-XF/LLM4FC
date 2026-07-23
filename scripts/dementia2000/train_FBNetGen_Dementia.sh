#!/bin/bash

python main.py --model "FBNetGen" --num_repeat 3 --dataset 'Dementia2000' --data_dir "../data/Dementia2000/Dementia2000.npy" --batch_size 16 --num_epochs 200 --drop_last False --mix_up --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test


