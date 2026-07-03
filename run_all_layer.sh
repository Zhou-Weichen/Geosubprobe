#!/bin/bash
set -e

MAX_LAYER=23
Optimizer=20_epoch # 10_epoch for nyuv2
CUDA=0

CUDA_VISIBLE_DEVICES=${CUDA} python train_depth.py probe=depth_linear backbone=dinov2_l14  optimizer=${Optimizer} +backbone.return_multilayer=True
CUDA_VISIBLE_DEVICES=${CUDA} python train_depth.py probe=depth_linear backbone=mae_l16     optimizer=${Optimizer} +backbone.return_multilayer=True
CUDA_VISIBLE_DEVICES=${CUDA} python train_depth.py probe=depth_linear backbone=ibot_l16    optimizer=${Optimizer} +backbone.return_multilayer=True

CUDA_VISIBLE_DEVICES=${CUDA} python train_depth.py probe=depth_dpt    backbone=dinov2_l14  optimizer=${Optimizer} +backbone.return_multilayer=True
CUDA_VISIBLE_DEVICES=${CUDA} python train_depth.py probe=depth_dpt    backbone=mae_l16     optimizer=${Optimizer} +backbone.return_multilayer=True
CUDA_VISIBLE_DEVICES=${CUDA} python train_depth.py probe=depth_dpt    backbone=ibot_l16    optimizer=${Optimizer} +backbone.return_multilayer=True

CUDA_VISIBLE_DEVICES=${CUDA} python train_depth.py probe=depth_mlp    backbone=dinov2_l14  optimizer=${Optimizer} +backbone.return_multilayer=True
CUDA_VISIBLE_DEVICES=${CUDA} python train_depth.py probe=depth_mlp    backbone=mae_l16     optimizer=${Optimizer} +backbone.return_multilayer=True
CUDA_VISIBLE_DEVICES=${CUDA} python train_depth.py probe=depth_mlp    backbone=ibot_l16    optimizer=${Optimizer} +backbone.return_multilayer=True


for ((i=0; i<=MAX_LAYER; i++)); do
  echo "========== Running layer ${i} =========="
  CUDA_VISIBLE_DEVICES=0  python train_depth.py \
    backbone=dinov2_l14  \
    optimizer=${Optimizer} \
    +backbone.return_multilayer=False \
    ++backbone.layer=${i} 
done

for ((i=0; i<=MAX_LAYER; i++)); do
  echo "========== Running layer ${i} =========="
  CUDA_VISIBLE_DEVICES=${CUDA}  python train_depth.py \
    backbone=mae_l16  \
    optimizer=${Optimizer} \
    +backbone.return_multilayer=False \
    ++backbone.layer=${i} 
done


for ((i=0; i<=MAX_LAYER; i++)); do
  echo "========== Running layer ${i} =========="
  CUDA_VISIBLE_DEVICES=${CUDA}  python train_depth.py \
    backbone=ibot_l16  \
    optimizer=${Optimizer} \
    +backbone.return_multilayer=False \
    ++backbone.layer=${i} 
done

