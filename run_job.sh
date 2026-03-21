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
#SBATCH -C RTX6000B|A100|V100
#SBATCH -p long
#SBATCH -t 48:00:00
# conda init 
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rl 
# python3 Wrapper.py --scene drone --checkpoint-root /home/apatwardhan/cv/nerf/checkpoint_long
python3 Wrapper.py --scene drone2 --checkpoint-root checkpoint_long \
  --epochs 100000 --val-interval 10 --save-interval 10 \
  --render-interval 10 --cache-images
# FOR PE 
# python3 reconstruct_best.py --scene drone --checkpoint-root checkpoint_long --run-tag pe_on --output-root recon_compare/normal
# FOR non PE
# python3 reconstruct_best.py --scene ship --checkpoint-root checkpoint_long --run-tag pe_on --output-root recon_compare/zero_pe --zero-pe-features 
# python3 Wrapper.py --scene ship --checkpoint-root checkpoint_long --run-tag no_pe --no-positional-encoding --epochs 100000 --val-interval 10 --save-interval 10 --render-interval 10 --cache-images
# RENDER
# python3 render.py --scene ship \
#   --checkpoint-root checkpoint_long \
#   --checkpoint-name nerf_model_epoch_013200.pth \
#   --output-root no_pe_2 \
