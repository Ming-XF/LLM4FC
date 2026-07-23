#!/bin/bash

python main.py --wandb_entity cwg --project Dementia --model "EEGChannelNet" --num_repeat 3 --dataset 'Dementia' --data_dir "../data/Dementia200/Dementia200.npy"  --batch_size 4 --num_epochs 200 --drop_last False --schedule cos --learning_rate 1e-3 --do_train --do_evaluate --do_test

