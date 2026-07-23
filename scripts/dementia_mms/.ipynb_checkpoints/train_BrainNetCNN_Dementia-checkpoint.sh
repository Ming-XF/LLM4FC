#!/bin/bash

python main.py --model "BrainNetCNN" --num_repeat 3 --dataset 'Dementia_MMS' --data_dir "../data/Dementia_MMS/Dementia_MMS.npz" --batch_size 16 --num_epochs 200 --drop_last False --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test

