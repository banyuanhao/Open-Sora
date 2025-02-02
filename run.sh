#!/bin/bash
#SBATCH --partition=hgx
#SBATCH --job-name=video
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time 72:00:00
#SBATCH --nodelist=h100-01    # 指定使用 h100-01 节点

torchrun --standalone --nproc_per_node 4 scripts/train_pixel.py configs/opensora-v1-2/train/pixel.py --data-path video_info.csv


torchrun --standalone --nproc_per_node 4 scripts/train_pixel.py configs/opensora-v1-2/train/pixel.py --data-path video_info.csv --load /data/xuchenheng/outputs/