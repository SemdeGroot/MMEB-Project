#!/bin/bash
#SBATCH -J mmeb-train
#SBATCH -t 04:00:00
#SBATCH -p gpu-short
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s3569829@umail.leidenuniv.nl
#SBATCH --output=logs/%j.out

# Usage:
#   bash job.sh                           submit all 7 experiments at once
#   sbatch --time=02:00:00 job.sh baseline resnet50   run one experiment

# When called with no arguments, submit all 7 jobs and exit.
if [ $# -eq 0 ]; then
    mkdir -p logs
    sbatch --time=04:00:00 -p gpu-short      "$0" location_only
    sbatch --time=04:00:00 -p gpu-short      "$0" baseline     resnet50
    sbatch --time=04:00:00 -p gpu-short      "$0" early_fusion resnet50
    sbatch --time=04:00:00 -p gpu-short      "$0" late_fusion  resnet50
    sbatch --time=06:00:00 -p gpu-2080ti-11g "$0" baseline     bioclip
    sbatch --time=10:00:00 -p gpu-2080ti-11g "$0" early_fusion bioclip
    sbatch --time=10:00:00 -p gpu-2080ti-11g "$0" late_fusion  bioclip
    echo "All 7 jobs submitted. Check status with: squeue --me"
    exit 0
fi

MODEL=${1:?usage: bash job.sh  OR  sbatch job.sh <model> [backbone]}
BACKBONE=${2:-resnet50}

WORKDIR=/zfsstore/courses/2025-2026/4343MMEBX/Group6
cd "$WORKDIR"
module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./env
export HF_HOME=/zfsstore/courses/2025-2026/4343MMEBX/Group6/.cache

mkdir -p logs

# download_images.py is idempotent: skips files that already exist
python data/download_images.py

if [ "$MODEL" = "location_only" ]; then
    python -m pipeline.train --model location_only --resume
    python -m pipeline.test  --model location_only
elif [ "$MODEL" = "baseline" ]; then
    python -m pipeline.train --model "$MODEL" --backbone "$BACKBONE" --resume
    python -m pipeline.test  --model "$MODEL" --backbone "$BACKBONE" --gradcam
else
    python -m pipeline.train --model "$MODEL" --backbone "$BACKBONE" --resume
    python -m pipeline.test  --model "$MODEL" --backbone "$BACKBONE"
fi
