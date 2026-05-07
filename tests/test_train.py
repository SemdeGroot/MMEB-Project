"""
Simulate how job.sh calls train.py on the real dataset, but with synthetic data
and 2 epochs so the test is fast. If this passes locally, the job on ALICE should work.

Run with:
    python tests/test_train.py
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

import pipeline.train

NUM_CLASSES = 5
N_SAMPLES   = 40


class FakeDataset(Dataset):
    """Synthetic stand-in for MothDataset. Returns random tensors of the right shape."""

    def __init__(self, n):
        species = [f"Moth species {i}" for i in range(NUM_CLASSES)]
        self.label_to_idx = {sp: i for i, sp in enumerate(species)}
        # samples mirrors MothDataset: (path, gbif_id, lat, lon, month, day, species)
        self.samples = [
            (Path("fake.jpg"), f"g{j}", 52.0, 5.0, 6, 15, species[j % NUM_CLASSES])
            for j in range(n)
        ]
        self._imgs   = torch.randn(n, 3, 224, 224)
        self._locs   = torch.randn(n, 4)
        self._labels = torch.randint(0, NUM_CLASSES, (n,))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._imgs[idx], self._locs[idx], self._labels[idx].item()


def make_fake_datasets(*_args, **_kwargs):
    ds = FakeDataset(N_SAMPLES)
    return ds, FakeDataset(10), FakeDataset(10), NUM_CLASSES, ds.label_to_idx


def run(model, backbone, out_dir, extra_args=None):
    argv = ["train.py", "--model", model, "--epochs", "2", "--batch_size", "4"]
    if backbone:
        argv += ["--backbone", backbone]
    if extra_args:
        argv += extra_args

    with (
        patch.object(pipeline.train, "get_datasets", side_effect=make_fake_datasets),
        patch.object(pipeline.train, "RESULTS_DIR",  out_dir),
        patch("sys.argv", argv),
    ):
        pipeline.train.main()


# All 9 experiments from job.sh
RUNS = [
    ("baseline",      "resnet50"),
    ("early_fusion",  "resnet50"),
    ("late_fusion",   "resnet50"),
    ("gated_fusion",  "resnet50"),
    ("location_only", None),
    ("baseline",      "bioclip"),
    ("early_fusion",  "bioclip"),
    ("late_fusion",   "bioclip"),
    ("gated_fusion",  "bioclip"),
]

import tempfile

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    for model, backbone in RUNS:
        label = f"{model}_{backbone}" if backbone else model
        print(f"\n--- {label} ---")
        run(model, backbone, tmp)

        out = tmp / (f"{model}_{backbone}" if backbone else "location_only")
        assert (out / "last.pt").exists(), "last.pt not written"
        assert (out / "best.pt").exists(), "best.pt not written"
        assert (out / "label_to_idx.json").exists(), "label_to_idx.json not written"

        ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        assert "model" in ckpt and "epoch" in ckpt
        assert ckpt["epoch"] == 1, f"Expected epoch 1 (last of 2), got {ckpt['epoch']}"

        with open(out / "label_to_idx.json") as f:
            mapping = json.load(f)
        assert len(mapping) == NUM_CLASSES

        print(f"  checkpoints OK, label_to_idx OK ({NUM_CLASSES} classes)")

    print("\n--- resume: baseline resnet50 ---")
    out = tmp / "baseline_resnet50"
    # First run already done above (epoch 0 and 1). Now resume for 2 more epochs (total 4).
    run("baseline", "resnet50", tmp, extra_args=["--resume", "--epochs", "4"])
    ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 3, f"Expected epoch 3 after resume, got {ckpt['epoch']}"
    print(f"  resumed correctly, last epoch={ckpt['epoch']}")

print("\nAll synthetic-data train tests passed")

# Real-data smoke test: use the actual pipeline but slice to a tiny subset so it's fast.
# Skipped automatically if no images have been downloaded yet.
from pipeline.preprocessing import IMAGES_DIR, get_datasets, MothDataset

n_images = len(list(IMAGES_DIR.glob("*/*.jpg")))
if n_images == 0:
    print("\nNo images found — skipping real-data test (run data/download_images.py first)")
else:
    print(f"\n--- real data smoke test ({n_images} images available) ---")

    REAL_N = 80  # enough to cover all NUM_CLASSES with some left for val/test

    def make_small_datasets(*_args, **kwargs):
        backbone = kwargs.get("backbone", "resnet50")
        train_ds, val_ds, test_ds, num_classes, label_to_idx = get_datasets(backbone=backbone)
        small_train = MothDataset(train_ds.samples[:REAL_N], label_to_idx, train_ds.transform)
        small_val   = MothDataset(val_ds.samples[:20],       label_to_idx, val_ds.transform)
        small_test  = MothDataset(test_ds.samples[:20],      label_to_idx, test_ds.transform)
        return small_train, small_val, small_test, num_classes, label_to_idx

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for model, backbone in [("baseline", "resnet50"), ("location_only", None)]:
            label = f"{model}_{backbone}" if backbone else model
            print(f"  {label} ...", end=" ", flush=True)
            argv = ["train.py", "--model", model, "--epochs", "2", "--batch_size", "4"]
            if backbone:
                argv += ["--backbone", backbone]
            with (
                patch.object(pipeline.train, "get_datasets", side_effect=make_small_datasets),
                patch.object(pipeline.train, "RESULTS_DIR", tmp),
                patch("sys.argv", argv),
            ):
                pipeline.train.main()
            out = tmp / (f"{model}_{backbone}" if backbone else "location_only")
            assert (out / "last.pt").exists()
            assert (out / "best.pt").exists()
            print("OK")

    print("Real-data smoke test passed")

print("\nDone")
