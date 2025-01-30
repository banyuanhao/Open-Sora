#!/bin/bash
#SBATCH --partition=hgx
#SBATCH --job-name=video
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time 48:00:00


torchrun --standalone --nproc_per_node 4 scripts/train.py \
    configs/opensora-v1-2/train/stage1.py --data-path video_info.csv --ckpt-path checkpoints