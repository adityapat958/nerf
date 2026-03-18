#!/bin/bash

#SBATCH --mail-user=apatwardhan@wpi.edu
#SBATCH --mail-type=ALL


#SBATCH -J NeRF_drone2
#SBATCH -o nerf_drone2_%j.out
#SBATCH -e nerf_drone2_%j.err
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C RTX6000B
#SBATCH -p long
#SBATCH -t 48:00:00
conda init
conda activate rl 
# python3 Wrapper.py --scene drone2 --checkpoint-root /home/apatwardhan/cv/nerf/checkpoint_long
python3 Wrapper.py --scene drone2 --checkpoint-root checkpoint_long --epochs 100000 --rays-per-batch 8192 --num-samples 64 --val-interval 10 --cache-images