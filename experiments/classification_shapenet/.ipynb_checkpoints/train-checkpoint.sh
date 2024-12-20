#!/usr/bin/env bash

device=2

diffusion_blocks=8
diffusion_features=64
k_eig=128
epochs=200

nohup python classification_shapenet.py \
        --input_features "xyz" \
        --diffusion_blocks ${diffusion_blocks} \
        --diffusion_features ${diffusion_features} \
        --n_class 16 \
        --k_eig ${k_eig} \
        --device ${device} \
        --lr 1e-4 \
        --epochs ${epochs} \
        --save_dir "saved_models_${diffusion_blocks}_blocks_${diffusion_features}_features" \
        --comet_experiment "Shapenet ${diffusion_blocks} blocks and ${diffusion_features} features " \
         > "output_${diffusion_blocks}blocks_${diffusion_features}feat.log" 2>&1 &
