#!/bin/bash

# 设置随机种子，确保可复现
export PYTHONHASHSEED=42
export CUDA_LAUNCH_BLOCKING=1

# 启动训练
python src/train.py \
    --model_type transformer \
    --dataset wikitext \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 0.0001 \
    --seed 42 \
    --save_dir ./results \
    --log_interval 100 \
    --device cuda

echo "Training completed! Results saved in ./results/"
