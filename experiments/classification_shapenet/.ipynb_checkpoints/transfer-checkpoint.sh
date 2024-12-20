#!/bin/bash

device=1

diffusion_blocks=8
diffusion_features=64
split_size=10
k_eig=128
epochs=200

nohup python transfer_shrec.py \
        --input_features "xyz" \
        --dataset_type "original" \
        --split_size ${split_size} \
        --diffusion_blocks ${diffusion_blocks} \
        --diffusion_features ${diffusion_features} \
        --k_eig ${k_eig} \
        --device ${device} \
        --lr 1e-5 \
        --save_dir "saved_models_${diffusion_blocks}_blocks_${diffusion_features}_features_transfer" \
        --epochs ${epochs} \
        --ckpt_path "saved_models_${diffusion_blocks}_blocks_${diffusion_features}_features/model_best_xyz.pth" \
        > "output_${diffusion_blocks}blocks_${diffusion_features}features_transfer.log" 2>&1 &
