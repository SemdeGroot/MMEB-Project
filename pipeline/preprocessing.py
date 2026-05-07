import csv
import math
import random
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


DATA_DIR   = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
OCCURRENCE = DATA_DIR / "occurrence.txt"

MIN_GBIFIDS = 10

# Netherlands bounding box that maps lat/lon to [-1, 1]
LAT_MIN, LAT_MAX = 50.7, 53.6
LON_MIN, LON_MAX = 3.3, 7.2

DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

STATE_PROVINCE_NORMALIZATION = {
    'Frisia':        'Friesland',
    'North Brabant': 'Noord-Brabant',
    'North Holland': 'Noord-Holland',
    'South Holland': 'Zuid-Holland',
}

# ResNet uses ImageNet stats; BioCLIP uses its own CLIP normalisation.
NORMALIZE_STATS = {
    "resnet50": ((0.485, 0.456, 0.406),     (0.229, 0.224, 0.225)),
    "bioclip":  ((0.48145466, 0.4578275, 0.40821073),
                 (0.26862954, 0.26130258, 0.27577711)),
}


def make_train_transform(backbone):
    mean, std = NORMALIZE_STATS[backbone]
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def make_val_transform(backbone):
    mean, std = NORMALIZE_STATS[backbone]
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ---- Raw data loading ----

def _safe_int(s):
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return None


def _safe_float(s):
    try:
        v = float(s.strip())
        return v if v >= 0 else None
    except (ValueError, AttributeError):
        return None


def _extract_hour(event_time_str):
    """Parse hour (float) from strings like '23:43:00+02:00'. Returns None if absent."""
    if not event_time_str:
        return None
    m = re.search(r'(\d{2}):(\d{2}):(\d{2})', event_time_str)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60.0
    return None


def _normalize_province(s):
    p = (s or "").strip()
    p = p if p else "Unknown"
    return STATE_PROVINCE_NORMALIZATION.get(p, p)


def load_occurrence_metadata():
    """Return dict mapping gbifID -> raw metadata dict with all needed fields."""
    metadata = {}
    with open(OCCURRENCE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gid     = row.get("gbifID", "").strip()
            species = row.get("scientificName", "").strip()
            lat     = row.get("decimalLatitude", "").strip()
            lon     = row.get("decimalLongitude", "").strip()
            month   = row.get("month", "").strip()
            day     = row.get("day", "").strip()
            if not (gid and species and lat and lon and month and day):
                continue
            try:
                metadata[gid] = {
                    "species":           species,
                    "lat":               float(lat),
                    "lon":               float(lon),
                    "month":             int(month),
                    "day":               int(day),
                    "year":              _safe_int(row.get("year", "")),
                    "event_time":        row.get("eventTime", "").strip(),
                    "state_province":    row.get("stateProvince", "").strip(),
                    "coord_uncertainty": _safe_float(
                        row.get("coordinateUncertaintyInMeters", "")),
                }
            except ValueError:
                continue
    return metadata


# ---- Metadata encoder (fitted on training set) ----

class MetadataEncoder:
    """Computes the full rich feature vector from a raw metadata dict.

    Call fit() on the training samples once, then encode_tensor() per sample.
    Fitting captures province vocabulary and year/uncertainty statistics so
    that val/test features stay on the same scale as training features.
    """

    def __init__(self):
        self.province_vocab   = []
        self.province_to_idx  = {}
        self.year_min         = None
        self.year_max         = None
        self.uncertainty_mean = None
        self.uncertainty_std  = None
        self.metadata_dim     = None

    def fit(self, raw_meta_list):
        provinces = sorted({_normalize_province(m["state_province"])
                            for m in raw_meta_list})
        self.province_vocab  = provinces
        self.province_to_idx = {p: i for i, p in enumerate(provinces)}

        years = [m["year"] for m in raw_meta_list if m["year"] is not None]
        if years:
            self.year_min = min(years)
            self.year_max = max(years)

        raw_u = [m["coord_uncertainty"] for m in raw_meta_list
                 if m["coord_uncertainty"] is not None]
        if raw_u:
            log_u = [math.log1p(max(0.0, u)) for u in raw_u]
            self.uncertainty_mean = float(np.mean(log_u))
            self.uncertainty_std  = float(np.std(log_u)) or 1.0

        self.metadata_dim = 17 + len(self.province_vocab)
        return self

    def encode_tensor(self, m):
        lat, lon, month, day = m["lat"], m["lon"], m["month"], m["day"]

        # Spatial features mapped to [-1, 1] using Netherlands bounding box
        lat_norm = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * 2 - 1
        lon_norm = (lon - LON_MIN) / (LON_MAX - LON_MIN) * 2 - 1

        # Cyclic month
        month_sin = math.sin(2 * math.pi * (month - 1) / 12)
        month_cos = math.cos(2 * math.pi * (month - 1) / 12)

        # Day of year (cyclic)
        doy     = sum(DAYS_PER_MONTH[:month - 1]) + day
        doy_sin = math.sin(2 * math.pi * doy / 365)
        doy_cos = math.cos(2 * math.pi * doy / 365)
        doy_missing = 0.0

        # Hour of day (cyclic)
        hour = _extract_hour(m["event_time"])
        hour_missing = 1.0 if hour is None else 0.0
        hour = 12.0 if hour is None else hour
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        # Interaction terms: space x time
        lat_month_interaction  = lat_norm * month_sin
        lat_season_interaction = lat_norm * month_cos
        lat_day_interaction    = lat_norm * doy_sin
        hour_month_interaction = hour_sin * month_sin

        # Observation year normalised to [0, 1]
        if (m["year"] is not None and self.year_min is not None
                and self.year_max is not None and self.year_max > self.year_min):
            year_norm = (m["year"] - self.year_min) / (self.year_max - self.year_min)
        else:
            year_norm = 0.0

        # Coordinate uncertainty (log-scale, z-scored)
        u = m["coord_uncertainty"]
        u_missing = 1.0 if u is None else 0.0
        if u is not None and self.uncertainty_mean is not None:
            u_log  = math.log1p(max(0.0, u))
            u_norm = (u_log - self.uncertainty_mean) / self.uncertainty_std
        else:
            u_norm = 0.0

        base = np.array([
            lat_norm, lon_norm,
            month_sin, month_cos,
            doy_sin, doy_cos,
            hour_sin, hour_cos,
            lat_month_interaction, lat_season_interaction,
            lat_day_interaction, hour_month_interaction,
            year_norm,
            doy_missing, hour_missing,
            u_norm, u_missing,
        ], dtype=np.float32)

        # Province one-hot shifted to [-0.5, 0.5]
        province = _normalize_province(m["state_province"])
        one_hot  = np.full(len(self.province_vocab), -0.5, dtype=np.float32)
        if province in self.province_to_idx:
            one_hot[self.province_to_idx[province]] = 0.5

        return torch.from_numpy(np.concatenate([base, one_hot]))


# ---- Dataset ----

class MothDataset(Dataset):
    def __init__(self, samples, label_to_idx, meta_encoder, transform=None):
        self.samples      = samples
        self.label_to_idx = label_to_idx
        self.meta_encoder = meta_encoder
        self.transform    = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, _gid, raw_meta, species = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        meta  = self.meta_encoder.encode_tensor(raw_meta)
        label = self.label_to_idx[species]
        return img, meta, label


# ---- Splitting (by gbifID to prevent data leakage) ----

def make_splits(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    by_species = defaultdict(list)
    for s in samples:
        by_species[s[-1]].append(s)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for items in by_species.values():
        by_gbifid = defaultdict(list)
        for s in items:
            by_gbifid[s[1]].append(s)

        gbif_ids = list(by_gbifid.keys())
        rng.shuffle(gbif_ids)

        n_train = int(len(gbif_ids) * train_ratio)
        n_val   = int(len(gbif_ids) * val_ratio)

        for gid in gbif_ids[:n_train]:
            train.extend(by_gbifid[gid])
        for gid in gbif_ids[n_train:n_train + n_val]:
            val.extend(by_gbifid[gid])
        for gid in gbif_ids[n_train + n_val:]:
            test.extend(by_gbifid[gid])

    return train, val, test


# ---- Entry point ----

def get_datasets(min_gbifids=MIN_GBIFIDS, backbone="resnet50"):
    """Build train/val/test datasets.

    Returns: train_ds, val_ds, test_ds, num_classes, label_to_idx, metadata_dim
    """
    metadata = load_occurrence_metadata()

    species_gbifids = defaultdict(set)
    all_samples     = []

    for img_path in IMAGES_DIR.glob("*/*.jpg"):
        gbif_id = img_path.stem.split("_")[0]
        if gbif_id not in metadata:
            continue
        m = metadata[gbif_id]
        all_samples.append((img_path, gbif_id, m, m["species"]))
        species_gbifids[m["species"]].add(gbif_id)

    allowed = {sp for sp, ids in species_gbifids.items() if len(ids) >= min_gbifids}
    samples = [s for s in all_samples if s[-1] in allowed]

    dropped_species = len(species_gbifids) - len(allowed)
    dropped_records = len(all_samples) - len(samples)
    print(f"Classes: {len(allowed)}, total samples: {len(samples)}")
    print(f"Discarded {dropped_species} species ({dropped_records} records) "
          f"below MIN_GBIFIDS={min_gbifids}")

    label_to_idx = {sp: i for i, sp in enumerate(sorted(allowed))}
    train_samples, val_samples, test_samples = make_splits(samples)

    # Fit the encoder on training data only so val/test use training statistics.
    meta_encoder = MetadataEncoder().fit([s[2] for s in train_samples])
    print(f"Metadata dim: {meta_encoder.metadata_dim} "
          f"(17 base + {len(meta_encoder.province_vocab)} provinces)")

    train_tf = make_train_transform(backbone)
    val_tf   = make_val_transform(backbone)

    train_ds = MothDataset(train_samples, label_to_idx, meta_encoder, transform=train_tf)
    val_ds   = MothDataset(val_samples,   label_to_idx, meta_encoder, transform=val_tf)
    test_ds  = MothDataset(test_samples,  label_to_idx, meta_encoder, transform=val_tf)

    return train_ds, val_ds, test_ds, len(allowed), label_to_idx, meta_encoder.metadata_dim
