#!/bin/bash
#SBATCH --partition=hgx
#SBATCH --job-name=video
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time 192:00:00
#SBATCH --nodelist=h100-02

torchrun --standalone --nproc_per_node 4 scripts/train_pixel.py configs/opensora-v1-2/train/pixel.py --data-path video_info.csv