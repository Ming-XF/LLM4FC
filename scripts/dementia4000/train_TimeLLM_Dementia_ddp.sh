#!/bin/bash
#
# TimeLLM DDP training on Dementia4000 (4 GPUs)
#
# Uses torchrun for multi-GPU DistributedDataParallel.
# Each GPU runs one process; DistributedSampler partitions the data.
#
# GPU memory: ChatGLM-6B ~12 GB + activations ~10 GB ≈ 22 GB per GPU
#   batch_size=1 per GPU for 24 GB cards; raise to 2 for 48 GB cards
#   Total effective batch size = 1 × 4 = 4

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 main.py \
    --model "TimeLLM" \
    --num_repeat 1 \
    --dataset 'Dementia4000' \
    --data_dir "../data/Dementia4000/Dementia4000.npz" \
    --batch_size 1 \
    --num_epochs 100 \
    --save_steps 25 \
    --drop_last False \
    --train_set 0.8 \
    --val_set 0.1 \
    --schedule 'cos' \
    --optimizer 'Adam' \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --eps 1e-8 \
    --warmup_steps 200 \
    --early_stop_patience 15 \
    --early_stop_min_delta 0.001 \
    --early_stop_metric "Accuracy" \
    --d_model 64 \
    --num_heads 8 \
    --patch_stride 1 \
    --num_prototypes 500 \
    --llm_layers 28 \
    --dropout 0.1 \
    --do_parallel \
    --do_train \
    --do_evaluate \
    --do_test
