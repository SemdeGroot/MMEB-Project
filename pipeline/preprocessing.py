import csv
import math
import random
from pathlib import Path
from collections import defaultdict

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


DATA_DIR   = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
OCCURRENCE = DATA_DIR / "occurrence.txt"

MIN_SAMPLES = 10  # species with fewer images are excluded

# Netherlands bounding box, used to normalize coordinates to [-1, 1]
LAT_MIN, LAT_MAX = 50.7, 53.6
LON_MIN, LON_MAX = 3.3, 7.2

DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_occurrence_metadata():
    """Read occurrence.txt and return a dict mapping gbif_id to (species, lat, lon, month, day)."""
    metadata = {}
    with open(OCCURRENCE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gid     = row.get("gbifID", "").strip()
            species = row.get("scientificName", "").strip()
            lat     = row.get("decimalLatitude", "").strip()
            lon     = row.get("decimalLongitude", "").strip()
            month   = row.get("month", "").strip()
            day     = row.get("day", "").strip()
            if gid and species and lat and lon and month and day:
                try:
                    metadata[gid] = (species, float(lat), float(lon), int(month), int(day))
                except ValueError:
                    continue
    return metadata


def encode_location(lat, lon, month, day):
    """Return a 4-element tensor: normalized lat, lon, sin and cos of day-of-year."""
    lat_norm = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * 2 - 1
    lon_norm = (lon - LON_MIN) / (LON_MAX - LON_MIN) * 2 - 1

    # sin/cos encoding so that late December and early January are close together
    doy = sum(DAYS_PER_MONTH[:month - 1]) + day
    sin_doy = math.sin(2 * math.pi * doy / 365)
    cos_doy = math.cos(2 * math.pi * doy / 365)

    return torch.tensor([lat_norm, lon_norm, sin_doy, cos_doy], dtype=torch.float32)


class MothDataset(Dataset):
    def __init__(self, samples, label_to_idx, transform=None):
        # samples is a list of (image_path, lat, lon, month, day, species)
        self.samples     = samples
        self.label_to_idx = label_to_idx
        self.transform   = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, _gid, lat, lon, month, day, species = self.samples[idx]

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        loc   = encode_location(lat, lon, month, day)
        label = self.label_to_idx[species]

        return img, loc, label


def make_splits(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split samples into train/val/test, stratified by species, splitting on gbifID."""
    by_species = defaultdict(list)
    for s in samples:
        by_species[s[-1]].append(s)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for items in by_species.values():
        # Group images by gbifID so all photos of one observation stay together
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


def get_datasets(min_samples=MIN_SAMPLES):
    """Build and return train, val, test datasets and the number of classes.

    Called from train.py. Scans the images directory, matches each image to its
    occurrence metadata, filters species below min_samples, and splits the data.
    """
    metadata = load_occurrence_metadata()

    # collect all samples that have both an image and valid metadata
    species_counts = defaultdict(int)
    all_samples = []

    for img_path in IMAGES_DIR.glob("*/*.jpg"):
        gbif_id = img_path.stem.split("_")[0]
        if gbif_id not in metadata:
            continue
        species, lat, lon, month, day = metadata[gbif_id]
        all_samples.append((img_path, gbif_id, lat, lon, month, day, species))
        species_counts[species] += 1

    # filter species that don't have enough samples
    allowed = {sp for sp, n in species_counts.items() if n >= min_samples}
    samples = [s for s in all_samples if s[-1] in allowed]

    dropped_species = len(species_counts) - len(allowed)
    dropped_records = len(all_samples) - len(samples)
    print(f"Classes: {len(allowed)}, total samples: {len(samples)}")
    print(f"Discarded {dropped_species} species ({dropped_records} records) "
          f"below MIN_SAMPLES={min_samples}")

    label_to_idx = {sp: i for i, sp in enumerate(sorted(allowed))}
    train_samples, val_samples, test_samples = make_splits(samples)

    train_ds = MothDataset(train_samples, label_to_idx, transform=train_transform)
    val_ds   = MothDataset(val_samples,   label_to_idx, transform=val_transform)
    test_ds  = MothDataset(test_samples,  label_to_idx, transform=val_transform)

    return train_ds, val_ds, test_ds, len(allowed), label_to_idx
