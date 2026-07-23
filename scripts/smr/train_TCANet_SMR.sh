#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=4

cd ../..
python main.py \
--wandb_entity cwg \
--project SMR \
\
--model "TCANet" \
--within_subject \
--num_repeat 5 \
--subject_num 9 \
\
--dataset SMR \
--data_dir "/data/datasets/SMR/SMR128.npy" \
--batch_size 32 \
--num_epochs 200 \
--drop_last True \
--schedule cos \
--learning_rate 1e-4 \
\
--do_train \
--do_evaluate \
--do_test


python main.py --wandb_entity cwg --project Dementia --model "TCANet" --num_repeat 3 --dataset 'Dementia' --data_dir "../data/Dementia200/Dementia200.npy" --batch_size 16 --num_epochs 200 --drop_last False --schedule cos --learning_rate 1e-4 --do_train --do_evaluate --do_test


# python main.py --wandb_entity cwg --project Dementia  --model "DFaST" --num_repeat 3 --dataset 'Dementia' --data_dir "../data/Dementia200/Dementia200.npy" --sparsity 0.6 --batch_size 16 --num_workers 5 --num_epochs 200 --frequency 128 --num_kernels 64 --window_size 32 --D 64 --p1 4 --p2 8 --drop_last False --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test
