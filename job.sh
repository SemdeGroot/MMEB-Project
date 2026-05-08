#!/bin/bash
#SBATCH -J mmeb-train
#SBATCH -t 12:00:00
#SBATCH -p gpu-2080ti-11g
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%j.out

# Nodes that have shown CUDA-init failures — jobs landing here fall back to CPU.
EXCLUDE_NODES="node860,node857"

# Usage:
#   bash job.sh                              submit all 16 experiments at once
#   sbatch job.sh baseline resnet50          run one experiment manually
#   sbatch job.sh metadata_only              run metadata-only model

# ---- Submit all jobs when called with no arguments ----
if [ $# -eq 0 ]; then
    mkdir -p logs

    # Pre-warm the BioCLIP cache so BioCLIP jobs don't race to download it.
    export HF_HOME=/zfsstore/courses/2025-2026/4343MMEBX/Group6/.cache
    if [ ! -d "$HF_HOME/hub/models--imageomics--bioclip" ]; then
        echo "Pre-warming BioCLIP cache (one-time download, ~330 MB)..."
        module load Miniconda3
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate ./env
        python -c "import open_clip; open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')"
    else
        echo "BioCLIP cache already present at $HF_HOME, skipping pre-warm."
    fi

    SB="sbatch --exclude=$EXCLUDE_NODES"

    # Metadata/location only (no image backbone, short)
    $SB --time=02:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" location_only
    $SB --time=02:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" metadata_only

    # ResNet-50 backbone (12h on 2080 Ti)
    $SB --time=12:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" baseline           resnet50
    $SB --time=12:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" early_fusion       resnet50
    $SB --time=12:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" late_fusion        resnet50
    $SB --time=12:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" gated_fusion       resnet50
    $SB --time=12:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" concat_fusion      resnet50
    $SB --time=12:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" transformer_fusion resnet50
    $SB --time=12:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" coordination       resnet50

    # BioCLIP backbone (20h on 2080 Ti — longer due to freeze/unfreeze and larger model)
    $SB --time=20:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" baseline           bioclip
    $SB --time=20:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" early_fusion       bioclip
    $SB --time=20:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" late_fusion        bioclip
    $SB --time=20:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" gated_fusion       bioclip
    $SB --time=20:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" concat_fusion      bioclip
    $SB --time=20:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" transformer_fusion bioclip
    $SB --time=20:00:00 -p gpu-2080ti-11g --gres=gpu:2080_ti:1 "$0" coordination       bioclip

    echo "All 16 jobs submitted (excluding: $EXCLUDE_NODES). Check: squeue --me"
    exit 0
fi

# ---- Single experiment (called by sbatch above) ----
MODEL=${1:?usage: bash job.sh  OR  sbatch job.sh <model> [backbone]}
BACKBONE=${2:-resnet50}

WORKDIR=/zfsstore/courses/2025-2026/4343MMEBX/Group6
cd "$WORKDIR"
module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./env
export HF_HOME=/zfsstore/courses/2025-2026/4343MMEBX/Group6/.cache

mkdir -p logs results

# download_image.py is idempotent — skips files that already exist
python data/download_images.py

if [ "$MODEL" = "location_only" ] || [ "$MODEL" = "metadata_only" ]; then
    python -m pipeline.train --model "$MODEL" --epochs 50 --batch_size 64 --resume
    python -m pipeline.test  --model "$MODEL"

elif [ "$MODEL" = "baseline" ]; then
    python -m pipeline.train --model "$MODEL" --backbone "$BACKBONE" --epochs 50 --batch_size 64 --resume
    python -m pipeline.test  --model "$MODEL" --backbone "$BACKBONE" --gradcam

else
    python -m pipeline.train --model "$MODEL" --backbone "$BACKBONE" --epochs 50 --batch_size 64 --resume
    python -m pipeline.test  --model "$MODEL" --backbone "$BACKBONE"
fi
