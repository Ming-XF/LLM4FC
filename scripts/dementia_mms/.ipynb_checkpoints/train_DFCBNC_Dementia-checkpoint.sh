#!/bin/bash

python main.py --model "DFCBNC" --num_repeat 3 --dataset 'Dementia_MMS' --data_dir "../data/XWDementia/Dementia_MMS.npy" --batch_size 16 --num_epochs 200 --drop_last False --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test

