#!/bin/bash
#SBATCH --job-name=mmeb
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1

# Usage:
#   sbatch --export=MODEL=early_fusion,BACKBONE=bioclip,PLOTS=1 test.slurm
#   sbatch --export=MODEL=location_only,BACKBONE=resnet50 test.slurm
#
# Set PLOTS=1 to also save confusion matrix and per-species accuracy

mkdir -p logs

source .venv/bin/activate

PLOTS_FLAG=""
if [ "${PLOTS:-0}" = "1" ]; then
    PLOTS_FLAG="--plots"
fi

python -m pipeline.test \
    --model      ${MODEL} \
    --backbone   ${BACKBONE:-bioclip} \
    --batch_size 64 \
    ${PLOTS_FLAG}
