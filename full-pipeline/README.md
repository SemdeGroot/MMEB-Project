# Multimodal Moth Classification

This project classifies moth species from GBIF data using image-only, metadata-only, and multimodal fusion models.

The main question is:

Can location and time metadata improve species classification compared with using moth images alone?

## Main Workflow

The current project is built around one training pipeline, a model package, and an analysis folder:

- [config.py](/Users/sara/PycharmProjects/MMEB_project/config.py)
- [dataset.py](/Users/sara/PycharmProjects/MMEB_project/dataset.py)
- [train.py](/Users/sara/PycharmProjects/MMEB_project/train.py)
- [analysis/explore_data.py](/Users/sara/PycharmProjects/MMEB_project/analysis/explore_data.py)
- [models/model_image.py](/Users/sara/PycharmProjects/MMEB_project/models/model_image.py)
- [models/model_metadata.py](/Users/sara/PycharmProjects/MMEB_project/models/model_metadata.py)
- [models/model_early_fusion.py](/Users/sara/PycharmProjects/MMEB_project/models/model_early_fusion.py)
- [models/model_concat_fusion.py](/Users/sara/PycharmProjects/MMEB_project/models/model_concat_fusion.py)
- [models/model_gated_fusion.py](/Users/sara/PycharmProjects/MMEB_project/models/model_gated_fusion.py)
- [models/model_transformer_fusion.py](/Users/sara/PycharmProjects/MMEB_project/models/model_transformer_fusion.py)
- [models/model_coordination.py](/Users/sara/PycharmProjects/MMEB_project/models/model_coordination.py)
- [models/model_cross_attention_transformer.py](/Users/sara/PycharmProjects/MMEB_project/models/model_cross_attention_transformer.py)
- [download_image.py](/Users/sara/PycharmProjects/MMEB_project/download_image.py)
- [analysis/fission_analysis.py](/Users/sara/PycharmProjects/MMEB_project/analysis/fission_analysis.py)

This README intentionally documents only the main training path and the files used for it.

## Dataset

Source: [GBIF](https://www.gbif.org/)

Required files:

```text
occurrence.txt
multimedia.txt
```

Optional local image cache:

```text
images/
```

The training pipeline uses:

- `species` as the label
- `decimalLatitude` and `decimalLongitude`
- `month`, `year`, `day`
- `eventTime`
- `stateProvince`
- `coordinateUncertaintyInMeters`
- `identifier` from `multimedia.txt` as image URL fallback

Only adult moths are kept during cleaning.

## Data Layout

The code reads paths from [config.py](/Users/sara/PycharmProjects/MMEB_project/config.py).

Expected structure:

```text
DATA_DIR/
|-- occurrence.txt
|-- multimedia.txt
`-- images/
```

Important:

- `DATA_DIR` and `IMAGE_DIR` in [config.py](/Users/sara/PycharmProjects/MMEB_project/config.py) should match the local machine.
- During training, the loader reads local images first.
- If a local image is missing, it falls back to the URL in `multimedia.txt`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Optional Image Download

To pre-download images before training:

```bash
python download_image.py
```

This saves images into `IMAGE_DIR` from [config.py](/Users/sara/PycharmProjects/MMEB_project/config.py).

## Training

Train one model at a time with [train.py](/Users/sara/PycharmProjects/MMEB_project/train.py):

```bash
python train.py --model image_only
python train.py --model metadata_only
python train.py --model early_fusion
python train.py --model concat_fusion
python train.py --model gated_fusion
python train.py --model transformer_fusion
python train.py --model coordination
python train.py --model cross_attention_tf
```

Available models:

- `image_only`: EfficientNet-B0 baseline on images
- `metadata_only`: MLP on metadata features
- `early_fusion`: concatenates image features and metadata before the final shared MLP
- `concat_fusion`: image and metadata embeddings are concatenated
- `gated_fusion`: learned gate balances image and metadata embeddings
- `transformer_fusion`: bidirectional cross-attention followed by a transformer encoder
- `coordination`: transformer-style fusion plus contrastive alignment loss between image and metadata embeddings
- `cross_attention_tf`: explicit cross-attention transformer fusion with contrastive-ready outputs

Each run:

- writes a log to `results/log_<model>.txt`
- saves the best checkpoint to `checkpoints/best_model_<model>.pth`
- overwrites the previous log and checkpoint for the same model name

## Analysis Scripts

For dataset inspection before training:

- [analysis/explore_data.py](/Users/sara/PycharmProjects/MMEB_project/analysis/explore_data.py)

Run:

```bash
python analysis/explore_data.py
```

This script:

- prints a detailed dataset overview to `results/explore_data/log_explore_data.txt`
- saves separate report figures in `results/explore_data/figures/`
- lists all available raw occurrence columns
- lists the raw metadata fields selected by this project
- lists the final processed metadata features used by the models

After training, the project also includes:

- [analysis/fission_analysis.py](/Users/sara/PycharmProjects/MMEB_project/analysis/fission_analysis.py)

Run fission analysis:

```bash
python analysis/fission_analysis.py
```

This script:

- loads all trained models from `checkpoints/`
- compares test accuracy across unimodal and multimodal models
- reports simple fission-style comparisons such as best unimodal vs best fusion
- evaluates the effect of removing image or metadata from the best fusion model

## Metadata Features

The current metadata pipeline in [dataset.py](/Users/sara/PycharmProjects/MMEB_project/dataset.py) includes:

- normalized latitude and longitude
- cyclic month features
- cyclic day-of-year features
- cyclic hour-of-day features
- simple interaction features between space and time
- normalized observation year
- missing-value indicators for date and time features
- normalized coordinate uncertainty
- missing-value indicator for coordinate uncertainty
- one-hot encoded `stateProvince` after basic name normalization

## Training Setup

Main training settings come from [config.py](/Users/sara/PycharmProjects/MMEB_project/config.py).

Current defaults:

- `NUM_SAMPLES = None`
- `MIN_SAMPLES_PER_CLASS = 20`
- `LIFE_STAGE = "Adult"`
- `IMAGE_SIZE = 224`
- `NUM_WORKERS = 8`
- `BATCH_SIZE = 256`
- `EPOCHS = 100`
- `LEARNING_RATE = 3e-5`
- `LABEL_SMOOTHING = 0.05` (kept in config for experiments)
- `TRAIN_SPLIT = 0.7`
- `VAL_SPLIT = 0.15`
- `TEST_SPLIT = 0.15`
- `RANDOM_SEED = 42`

The training loop uses:

- focal loss
- Adam optimizer
- `ReduceLROnPlateau` scheduler
- deterministic seeding setup
- slightly different learning rates for image-only, metadata-only, and fusion runs

## Outputs

The project creates these directories automatically if they do not exist:

```text
checkpoints/
results/
```

Typical outputs:

- `checkpoints/best_model_image_only.pth`
- `checkpoints/best_model_transformer_fusion.pth`
- `results/log_image_only.txt`
- `results/log_transformer_fusion.txt`
- `results/explore_data/log_explore_data.txt`
- `results/explore_data/figures/geographic_distribution.png`
