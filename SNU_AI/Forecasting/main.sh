#!/bin/bash


for lr in 1e-4 1e-5 1e-6
do
    for min_lr in 1e-6 1e-7 1e-8 1e-9
    do
        python train.py \
            --gpu_id=0 \
            --data_dir=./data/SoC_synthesis \
            --lambda1=0.1 --lr=$lr --min_lr=$min_lr \
            --save_path=./data/train_result/lr$lr\_min_lr$min_lr
    done
done
