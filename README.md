# MMEB

This repository contains the code for Group 6's project in the course Multimodal Models for Ecology and Biodiversity. We classify Dutch moth species from GBIF observation data and test whether adding GPS coordinates and timestamps to image features improves accuracy over an image-only baseline (ResNet50 and BioCLIP).

## Project structure

The `models/` folder contains the five model definitions. `baseline` is image-only (`resnet50` or `bioclip`), `location_only` uses GPS and date only (MLP), and `early_fusion`, `late_fusion`, and `gated_fusion` combine both modalities.

The `pipeline/` folder contains three scripts. `preprocessing.py` defines the Dataset class, the train/val/test split on `gbifID`, and the 4-element location encoding. `train.py` runs the training loop with weighted cross-entropy, cosine learning rate, early stopping, and checkpointing to `best.pt` and `last.pt`. It supports a `--resume` flag. `test.py` loads `best.pt`, computes macro-F1 and per-bucket F1, writes `metrics.json`, and saves Grad-CAM overlays for the baseline image models.

The `analysis/` folder contains two scripts. `plot_data_description.py` produces dataset statistics and figures (samples per species, monthly distribution, geographic distribution). `evaluate_results.py` reads `metrics.json` from each result folder and produces cross-model comparison plots and .tex tables.

The `data/` folder holds the GBIF source files (`occurrence.txt`, `multimedia.txt`) and `download_images.py`. The `tests/` folder contains unit tests that mirror the real pipeline. The `results/` folder contains one subfolder per trained model with `best.pt`, `last.pt`, `metrics.json`, `history.json`, and an optional `gradcam/` folder for the baseline image models.

The `job.sh` script is the SLURM batch script for ALICE. When called without arguments, it submits all nine experiments in parallel.

## Setup

```bash
conda env create -f environment.yml -p ./env
conda activate ./env
```

## Download images

```bash
python data/download_images.py
```

The script skips files that already exist, so re-running after a crash is safe.

## Train one model

```bash
python -m pipeline.train --model baseline --backbone resnet50
python -m pipeline.train --model location_only
```

## Test a trained model

```bash
python -m pipeline.test --model baseline --backbone resnet50 --gradcam
```

The `metrics.json` file is always written. The `--gradcam` flag saves overlays for the baseline image models.

## Run the analysis

```bash
python -m analysis.plot_data_description
python -m analysis.evaluate_results
```

Outputs go to `analysis/output/`.

## Run all experiments on ALICE

```bash
bash job.sh
```

This submits nine SLURM jobs in parallel, one per model and backbone combination. See `CLAUDE.md` for SLURM details, conda setup on ALICE, and the list of excluded nodes.

## Design choices

For the rationale behind the model architectures, hyperparameters (loss, learning rates, freeze schedule, augmentation), and dataset decisions (species filter, gbifID-based splitting, sqrt-inverse class weights), see our report.

## Other branches

The `merged` branch contains an earlier experiment by a teammate that added FocalLoss, transformer-based fusion, and a richer metadata encoder. The README on that branch explains why those changes did not move into `main`.