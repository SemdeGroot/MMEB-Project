import os
import random
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import config

# Small cleanup map for province names that appear with English and Dutch variants.
STATE_PROVINCE_NORMALIZATION = {
    'Frisia': 'Friesland',
    'North Brabant': 'Noord-Brabant',
    'North Holland': 'Noord-Holland',
    'South Holland': 'Zuid-Holland',
}


# ---- Step 1: Load and merge occurrence + multimedia ----
def _get_local_image_path(gbif_id):
    # Images are stored locally as <gbifID>.jpg when pre-downloaded.
    return os.path.join(config.IMAGE_DIR, f"{gbif_id}.jpg")


def load_data():
    print("Loading occurrence data...")
    occ = pd.read_csv(config.OCCURRENCE_FILE, sep='\t', low_memory=False,
                      usecols=['gbifID', 'species', 'decimalLatitude',
                                'decimalLongitude', 'month', 'year', 'day',
                                'lifeStage', 'eventTime', 'stateProvince',
                                'coordinateUncertaintyInMeters'])

    print("Loading multimedia data...")
    med = pd.read_csv(config.MULTIMEDIA_FILE, sep='\t', low_memory=False,
                      usecols=['gbifID', 'identifier', 'format'])

    # Keep only jpeg images
    med['format'] = med['format'].astype(str).str.lower()
    med = med[med['format'] == 'image/jpeg']

    # Keep first image per occurrence
    med = med.groupby('gbifID').first().reset_index()

    # Merge
    df = pd.merge(occ, med, on='gbifID', how='inner')

    # Add local image path column
    df['local_path'] = df['gbifID'].apply(_get_local_image_path)

    # Prefer the local cache during training and fall back to the URL otherwise.
    df['image_source'] = df.apply(
        lambda row: row['local_path']
        if os.path.exists(row['local_path'])
        else row['identifier'], axis=1
    )
    df['image_source_type'] = np.where(
        df['image_source'] == df['local_path'],
        'local',
        'url'
    )

    local_count = int((df['image_source_type'] == 'local').sum())
    url_count = int((df['image_source_type'] == 'url').sum())
    print(f"Image sources: {local_count} local | {url_count} URL fallback")

    return df


# ---- Step 2: Clean and filter ----
def clean_data(df):
    print(f"Before cleaning: {len(df)} samples")

    # Filter adults only
    df = df[df['lifeStage'] == config.LIFE_STAGE]

    # Drop rows with missing values in key columns
    df = df.dropna(subset=['species', 'decimalLatitude', 'decimalLongitude',
                            'month', 'identifier'])

    # Drop species with too few samples
    species_counts = df['species'].value_counts()
    valid_species = species_counts[species_counts >= config.MIN_SAMPLES_PER_CLASS].index
    df = df[df['species'].isin(valid_species)]

    # Optional: limit total samples (for quick testing)
    if config.NUM_SAMPLES is not None:
        sample_size = min(config.NUM_SAMPLES, len(df))
        df = df.sample(n=sample_size, random_state=config.RANDOM_SEED)

    df = df.reset_index(drop=True)
    print(f"After cleaning: {len(df)} samples, {df['species'].nunique()} species")

    return df


# ---- Step 3: Encode labels ----
def encode_labels(df):
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['species'])
    return df, le


# ---- Step 4: Split ----
def split_data(df):
    use_stratify = df['label'].value_counts().min() >= 2

    train_df, temp_df = train_test_split(df, test_size=(1 - config.TRAIN_SPLIT),
                                         random_state=config.RANDOM_SEED,
                                         stratify=df['label'] if use_stratify else None)
    val_size = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_size),
                                       random_state=config.RANDOM_SEED,
                                       stratify=temp_df['label'] if use_stratify else None)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---- Step 5: Metadata preprocessing ----
def _encode_cyclic(series, period, zero_based=False):
    # Map cyclic values such as month or hour onto a circle.
    values = pd.to_numeric(series, errors='coerce')
    fill_value = values.median()
    if pd.isna(fill_value):
        fill_value = 0.0
    values = values.fillna(fill_value)
    if zero_based:
        angle = 2 * np.pi * values / period
    else:
        angle = 2 * np.pi * (values - 1) / period
    return np.sin(angle), np.cos(angle)


def _extract_event_hour(event_time_series):
    # Extract the hour component from strings such as 23:43:00+02:00.
    time_text = event_time_series.fillna('').astype(str).str.extract(r'(\d{2}:\d{2}:\d{2})', expand=False)
    time_delta = pd.to_timedelta(time_text, errors='coerce')
    hours = time_delta.dt.total_seconds() / 3600.0
    fill_value = hours.median()
    if pd.isna(fill_value):
        fill_value = 12.0
    return hours.fillna(fill_value)


def get_metadata_columns(df):
    state_cols = sorted(col for col in df.columns if col.startswith('state_'))
    base_cols = [
        'lat_norm', 'lon_norm',
        'month_sin', 'month_cos',
        'day_of_year_sin', 'day_of_year_cos',
        'hour_sin', 'hour_cos',
        'lat_month_interaction',
        'lat_season_interaction',
        'lat_day_interaction',
        'hour_month_interaction',
        'year_norm',
        'day_of_year_missing',
        'event_time_missing',
        'coord_uncertainty_norm',
        'coord_uncertainty_missing',
    ]
    return [col for col in base_cols if col in df.columns] + state_cols


def _normalize_state_province(series):
    # Standardise blanks and known duplicate spellings before one-hot encoding.
    province = series.fillna('Unknown').astype(str).str.strip()
    province = province.where(province.ne(''), 'Unknown')
    province = province.replace(STATE_PROVINCE_NORMALIZATION)
    return province


def preprocess_metadata(df):
    df = df.copy()

    # Keep spatial features on a comparable scale.
    df['lat_norm'] = df['decimalLatitude'] / 90.0
    df['lon_norm'] = df['decimalLongitude'] / 180.0

    # Month is cyclic, so December and January should stay close.
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Keep finer seasonal information without dropping the original month signal.
    dates = pd.to_datetime(
        dict(
            year=pd.to_numeric(df['year'], errors='coerce'),
            month=pd.to_numeric(df['month'], errors='coerce'),
            day=pd.to_numeric(df['day'], errors='coerce'),
        ),
        errors='coerce'
    )
    df['day_of_year_missing'] = dates.isna().astype(np.float32)
    day_of_year = dates.dt.dayofyear.fillna(183)
    df['day_of_year_sin'], df['day_of_year_cos'] = _encode_cyclic(day_of_year, period=366)

    # Time of day can help for nocturnal activity patterns.
    event_hours = _extract_event_hour(df['eventTime'])
    df['event_time_missing'] = pd.to_timedelta(
        df['eventTime'].fillna('').astype(str).str.extract(r'(\d{2}:\d{2}:\d{2})', expand=False),
        errors='coerce'
    ).isna().astype(np.float32)
    df['hour_sin'], df['hour_cos'] = _encode_cyclic(event_hours, period=24, zero_based=True)

    # Province is kept as one-hot metadata after light name cleanup.
    province = _normalize_state_province(df['stateProvince'])
    province_dummies = pd.get_dummies(province, prefix='state', dtype=float)
    # Shift one-hot values closer to the scale of the continuous features.
    province_dummies = province_dummies - 0.5
    df = pd.concat([df, province_dummies], axis=1)

    # Coordinate uncertainty spans a wide range, so log-scaling is more stable.
    uncertainty = pd.to_numeric(df['coordinateUncertaintyInMeters'], errors='coerce').clip(lower=0)
    df['coord_uncertainty_missing'] = uncertainty.isna().astype(np.float32)
    uncertainty = np.log1p(uncertainty)
    uncertainty = uncertainty.fillna(uncertainty.median() if uncertainty.notna().any() else 0.0)
    std = uncertainty.std()
    df['coord_uncertainty_norm'] = (
        (uncertainty - uncertainty.mean()) / std if std and not np.isnan(std) else uncertainty - uncertainty.mean()
    )

    # Simple interaction terms let the metadata model express space-time effects.
    df['lat_month_interaction'] = df['lat_norm'] * df['month_sin']
    df['lat_season_interaction'] = df['lat_norm'] * df['month_cos']
    df['lat_day_interaction']    = df['lat_norm'] * df['day_of_year_sin']

    df['hour_month_interaction'] = df['hour_sin'] * df['month_sin']

    # Observation year is scaled to [0, 1] when it varies in the dataset.
    if 'year' in df.columns:
        years = pd.to_numeric(df['year'], errors='coerce')
        year_min = years.min()
        year_max = years.max()
        if year_max > year_min:
            df['year_norm'] = (years - year_min) / (year_max - year_min)
        else:
            df['year_norm'] = 0.0

    return df


# ---- Step 6: PyTorch Dataset ----
class MothDataset(Dataset):
    def __init__(self, df, transform=None, meta_cols=None):
        self.df = df
        self.transform = transform
        self.meta_cols = meta_cols or get_metadata_columns(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = self._load_image(row['image_source'])

        # Metadata tensor
        metadata = torch.tensor(row[self.meta_cols].values.astype(np.float32))

        # Label
        label = torch.tensor(row['label'], dtype=torch.long)

        return image, metadata, label

    def _load_image(self, image_source):
        try:
            if isinstance(image_source, str) and image_source.startswith(('http://', 'https://')):
                timeout = getattr(config, 'IMAGE_DOWNLOAD_TIMEOUT', 10)
                response = requests.get(image_source, timeout=timeout)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_source).convert('RGB')
        except Exception:
            # Keep the batch shape valid even when a file is missing or broken.
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE))

        if self.transform:
            image = self.transform(image)
        return image


# ---- Step 7: Transforms ----
def get_transforms(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])


# ---- Step 8: DataLoaders ----
def _seed_worker(worker_id):
    # Re-seed each worker so shuffling and augmentations stay reproducible.
    worker_seed = config.RANDOM_SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_dataloaders(train_df, val_df, test_df):
    meta_cols = get_metadata_columns(train_df)
    train_ds = MothDataset(train_df, transform=get_transforms('train'), meta_cols=meta_cols)
    val_ds   = MothDataset(val_df,   transform=get_transforms('val'),   meta_cols=meta_cols)
    test_ds  = MothDataset(test_df,  transform=get_transforms('test'),  meta_cols=meta_cols)

    generator = torch.Generator()
    generator.manual_seed(config.RANDOM_SEED)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=config.NUM_WORKERS,
                              worker_init_fn=_seed_worker, generator=generator)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              worker_init_fn=_seed_worker, generator=generator)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              worker_init_fn=_seed_worker, generator=generator)

    return train_loader, val_loader, test_loader


# ---- Main: run to verify pipeline ----
if __name__ == '__main__':
    df = load_data()
    df = clean_data(df)
    df = preprocess_metadata(df)
    df, label_encoder = encode_labels(df)
    train_df, val_df, test_df = split_data(df)

    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)

    # Quick sanity check for shapes and metadata statistics.
    images, metadata, labels = next(iter(train_loader))
    print(f"\nBatch check:")
    print(f"  Images:   {images.shape}")
    print(f"  Metadata: {metadata.shape}")
    print(f"  Labels:   {labels.shape}")
    print(f"\nNum classes: {df['species'].nunique()}")
    print(f"Device: {config.DEVICE}")

    import sys
    meta_cols = get_metadata_columns(df)

    os.makedirs("results", exist_ok=True)
    log_path = os.path.join("results", "log_dataset_stats.txt")
    log_file = open(log_path, "w")

    def _print(msg=""):
        print(msg)
        log_file.write(msg + "\n")

    _print("\n" + "=" * 65)
    _print("METADATA STATISTICS")
    _print("=" * 65)
    for col in meta_cols:
        if col in df.columns:
            _print(f"{col:35} min={df[col].min():8.4f}  max={df[col].max():8.4f}  "
                   f"mean={df[col].mean():8.4f}  std={df[col].std():8.4f}")

    nan_count = df[meta_cols].isna().sum()
    if nan_count.sum() > 0:
        _print("\nWARNING: NaN values detected in metadata:")
        _print(str(nan_count[nan_count > 0]))
    else:
        _print("\nNo NaN values detected in metadata.")

    log_file.close()
    print(f"\nDataset stats saved to {log_path}")
