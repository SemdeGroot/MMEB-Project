#!/bin/bash
#SBATCH --job-name=mmeb
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1

# Usage:
#   sbatch --export=MODEL=early_fusion,BACKBONE=bioclip train.slurm
#   sbatch --export=MODEL=location_only,BACKBONE=resnet50 train.slurm
#
# MODEL choices:
#   baseline, location_only, metadata_only,
#   early_fusion, late_fusion, gated_fusion,
#   concat_fusion, transformer_fusion, coordination
#
# BACKBONE choices: resnet50, bioclip
# (location_only and metadata_only ignore --backbone)

mkdir -p logs results

source .venv/bin/activate

python -m pipeline.train \
    --model      ${MODEL} \
    --backbone   ${BACKBONE:-bioclip} \
    --epochs     50 \
    --batch_size 64
