# MMEB Project

This repository contains the training, testing, and analysis code for moth species classification with images and observation metadata.

The data in this branch is based on GBIF Netherlands records. The pipeline uses downloaded moth images together with observation metadata from `data/occurrence.txt`.

## Project layout

- `models/`: model definitions
- `pipeline/`: training, testing, and preprocessing
- `analysis/`: data exploration and result analysis
- `data/`: GBIF files and image download script

## Models

This branch includes:

- `baseline`: image-only classifier
- `location_only`: shallow metadata-only baseline, in this branch it uses the same full metadata vector as `metadata_only`
- `metadata_only`: deeper metadata-only classifier that uses the same full metadata vector as `location_only`
- `early_fusion`: encodes metadata with a small MLP and concatenates it with image features before classification
- `late_fusion`: uses separate image and metadata heads and sums their logits
- `gated_fusion`: learns a gate that mixes projected image and metadata features before classification
- `concat_fusion`: projects image and metadata into the same embedding size, concatenates them, and classifies the fused vector
- `transformer_fusion`: projects image and metadata into tokens, applies cross-attention, then classifies the fused token
- `coordination`: transformer-based fusion model that adds a contrastive alignment loss between image and metadata embeddings

Image-based models support two backbones:

- `resnet50`
- `bioclip`

## Data and preprocessing

Main input files:

- `data/occurrence.txt`
- `data/images/`

The preprocessing code is in `pipeline/preprocessing.py`. It:

- loads metadata from `occurrence.txt`
- normalizes province names
- builds image and metadata samples
- filters species with too few GBIF IDs
- splits data into train, validation, and test sets
- fits metadata scaling on the training split only

The current metadata includes:

- latitude and longitude
- month sine and cosine features
- day-of-year sine and cosine features
- hour sine and cosine features from `eventTime`
- year
- coordinate uncertainty
- interaction terms between space and time features
- normalized `stateProvince` features

Both `location_only` and `metadata_only` receive this same metadata vector. The difference between them is model capacity, not which metadata fields are used.

In the current branch, the metadata encoder builds:

- 17 base features
- one shifted one-hot feature per province in the training split
- metadata dimension = `17 + number_of_provinces`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download images

```bash
python data/download_images.py
```

## Run overview

Typical order:

- download images
- train a model
- test the trained model
- run the analysis files on the saved results

### Default training setup

The current training defaults in `pipeline/train.py` are:

- epochs: `50`
- batch size: `64`
- optimizer: `Adam`
- loss: `FocalLoss`
- early stopping patience: `5`
- BioCLIP freeze period: first `5` epochs

### Train

```bash
python -m pipeline.train --model baseline --backbone resnet50 --epochs 50 --batch_size 64
python -m pipeline.train --model location_only --epochs 50 --batch_size 64
```

### Test

```bash
python -m pipeline.test --model baseline --backbone resnet50 --batch_size 64 --plots
python -m pipeline.test --model location_only --batch_size 64 --plots
```

## Analysis

Available analysis files:

- `python -m analysis.explore_data`: prints dataset counts and filtering steps, and saves a log plus exploration figures in `results/explore_data/`
- `python -m analysis.plot_data_description`: saves `obs_per_species.png`, `monthly_distribution.png`, and `geographic_distribution.png` in `analysis/output/`
- `python -m analysis.modality_contribution_analysis`: compares unimodal and fusion runs, runs missing-modality tests, and saves `log_modality_contribution_analysis.txt` in `analysis/output/modality_contribution/`
- `python -m analysis.evaluate_results`: builds `macro_f1_comparison.png`, `bucket_f1_comparison.png`, `location_gain.png`, `bucket_gain.png`, `learning_curves.png`, `main_results.tex`, and `gain_table.tex` in `analysis/output/`

## Outputs

Training and test outputs are saved in `results/`.

- metadata-only models save to `results/<model_name>/`
- image and fusion models save to `results/<model_name>_<backbone>/`

Typical files in each result folder:

- `best.pt`
- `last.pt`
- `history.json`
- `label_to_idx.json`
- `metrics.json`
- `plots/` if test is run with `--plots`
- `gradcam/` if test is run with `--gradcam` for an image-based model

The optional test plots are:

- `plots/confusion_matrix.png`
- `plots/per_species_accuracy.png`

These plots are useful as diagnostic checks, but they are not very useful for the report because the task has many species and the full plots become too dense to read clearly.

Analysis outputs are saved in `analysis/output/` and data exploration outputs are saved in `results/explore_data/`.
