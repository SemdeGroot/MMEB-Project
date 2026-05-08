#!/bin/bash
#SBATCH --job-name=mmeb-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu-2080ti-11g
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Usage examples:
#   sbatch --export=MODEL=baseline,BACKBONE=resnet50 train.slurm
#   sbatch --export=MODEL=metadata_only train.slurm
#   sbatch --export=MODEL=coordination,BACKBONE=bioclip train.slurm

MODEL=${MODEL:?MODEL is required}
BACKBONE=${BACKBONE:-resnet50}

mkdir -p logs results

WORKDIR=/home/s4609808/projects/MMEB-Project
cd "$WORKDIR"
module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./env
export HF_HOME=$HOME/.cache/huggingface

python data/download_images.py

if [ "$MODEL" = "location_only" ] || [ "$MODEL" = "metadata_only" ]; then
    python -m pipeline.train \
        --model      "$MODEL" \
        --epochs     50 \
        --batch_size 64 \
        --resume
else
    python -m pipeline.train \
        --model      "$MODEL" \
        --backbone   "$BACKBONE" \
        --epochs     50 \
        --batch_size 64 \
        --resume
fi
