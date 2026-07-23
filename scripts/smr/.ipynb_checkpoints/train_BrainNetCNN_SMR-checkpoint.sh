#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=3



cd ../..
python main.py \
\
--model "BrainNetCNN" \
--within_subject \
--num_repeat 5 \
--subject_num 9 \
\
--dataset 'SMR' \
--data_dir "../data/SMR/SMR128.npy" \
--batch_size 32 \
--num_epochs 200 \
--drop_last True \
--schedule 'cos' \
--learning_rate 1e-3 \
\
--do_train \
--do_evaluate \
--do_test

python main.py --model "BrainNetCNN" --num_repeat 3 --dataset 'Dementia' --data_dir "../data/Dementia200/Dementia200.npy" --batch_size 16 --num_epochs 200 --drop_last False --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test

