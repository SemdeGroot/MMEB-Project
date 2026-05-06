import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import load_occurrence_metadata, encode_location, make_splits, get_datasets, IMAGES_DIR
import torch

print("--- metadata loading ---")
metadata = load_occurrence_metadata()
print(f"Records loaded: {len(metadata)}")
assert len(metadata) > 0, "No records loaded"

print("\n--- encode_location ---")
_, (species, lat, lon, month, day) = list(metadata.items())[0]
loc = encode_location(lat, lon, month, day)
assert loc.shape == torch.Size([4])
assert loc.dtype == torch.float32
assert -1.0 <= loc[0].item() <= 1.0, "lat not normalized to [-1, 1]"
assert -1.0 <= loc[1].item() <= 1.0, "lon not normalized to [-1, 1]"
assert -1.0 <= loc[2].item() <= 1.0, "sin_doy out of range"
assert -1.0 <= loc[3].item() <= 1.0, "cos_doy out of range"
print(f"Location tensor: {loc}")

print("\n--- make_splits (real metadata, no images) ---")
# Build fake sample tuples from real metadata to test splitting logic
samples = [
    (Path("fake.jpg"), gid, lat, lon, month, day, sp)
    for gid, (sp, lat, lon, month, day) in list(metadata.items())[:300]
]
train, val, test = make_splits(samples)
print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
assert len(train) + len(val) + len(test) == len(samples)

train_ids = {s[1] for s in train}
val_ids   = {s[1] for s in val}
test_ids  = {s[1] for s in test}
assert train_ids.isdisjoint(val_ids),  "gbifID leakage between train and val"
assert train_ids.isdisjoint(test_ids), "gbifID leakage between train and test"
assert val_ids.isdisjoint(test_ids),   "gbifID leakage between val and test"
print("No gbifID leakage between splits")

print("\n--- get_datasets (requires images) ---")
n_images = len(list(IMAGES_DIR.glob("*/*.jpg")))
if n_images == 0:
    print("No images found, skipping (run download_images.py first)")
else:
    print(f"Found {n_images} images, running get_datasets...")
    train_ds, val_ds, test_ds, n_classes, label_to_idx = get_datasets()
    print(f"Classes: {n_classes}, train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    assert n_classes > 0
    assert len(train_ds) > 0

    img, loc, label = train_ds[0]
    assert img.shape == (3, 224, 224), f"Unexpected image shape: {img.shape}"
    assert loc.shape[0] == 4
    assert 0 <= label < n_classes
    print(f"Sample: img={img.shape}, loc={loc}, label={label}")

print("\nAll dataset tests passed")
