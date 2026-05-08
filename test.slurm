#!/bin/bash
#SBATCH --job-name=mmeb-test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu-2080ti-11g
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Usage examples:
#   sbatch --export=MODEL=baseline,BACKBONE=resnet50,PLOTS=1 test.slurm
#   sbatch --export=MODEL=metadata_only test.slurm
#   sbatch --export=MODEL=transformer_fusion,BACKBONE=bioclip,PLOTS=1 test.slurm

MODEL=${MODEL:?MODEL is required}
BACKBONE=${BACKBONE:-resnet50}

mkdir -p logs
WORKDIR=/home/s4609808/projects/MMEB-Project
cd "$WORKDIR"

module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./env
export HF_HOME=$HOME/.cache/huggingface

PLOTS_FLAG=""
if [ "${PLOTS:-0}" = "1" ]; then
    PLOTS_FLAG="--plots"
fi

if [ "$MODEL" = "location_only" ] || [ "$MODEL" = "metadata_only" ]; then
    python -m pipeline.test \
        --model      "$MODEL" \
        --batch_size 64 \
        ${PLOTS_FLAG}
else
    python -m pipeline.test \
        --model      "$MODEL" \
        --backbone   "$BACKBONE" \
        --batch_size 64 \
        ${PLOTS_FLAG}
fi
