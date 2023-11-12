#!/usr/bin/bash

#SBATCH -J train_run_epoch10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -w aurora-g7
#SBATCH -o logs/slurm-%A.out

tar xzvf flying.tar.gz -C /local_datasets/flying/
cd /data/operati123/yolo
python inference.py

