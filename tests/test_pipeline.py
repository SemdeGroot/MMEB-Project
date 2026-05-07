"""
End-to-end simulation of job.sh using real data (small subset).
Covers: image download check → dataset build → train → evaluate.
If this passes locally, job.sh should work on ALICE without surprises.

Run with:
    python tests/test_pipeline.py
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pipeline.train
import pipeline.test as pipeline_test
from pipeline.preprocessing import IMAGES_DIR, MothDataset, get_datasets

MIN_IMAGES = 50
SMALL_TRAIN = 80
SMALL_VALTEST = 20

# Step 1: ensure enough images exist (same threshold logic as job.sh)
n_images = len(list(IMAGES_DIR.glob("*/*.jpg")))
if n_images < MIN_IMAGES:
    print(f"Only {n_images} images found — downloading to reach {MIN_IMAGES}...")
    from test_download_images import ensure_sample_images
    ensure_sample_images()
    n_images = len(list(IMAGES_DIR.glob("*/*.jpg")))

print(f"Images available: {n_images}")
assert n_images >= MIN_IMAGES, f"Need at least {MIN_IMAGES} images"

# Step 2: build a small slice of the real dataset once, reused for all runs
print("\n--- building datasets ---")
train_ds_full, val_ds_full, test_ds_full, num_classes, label_to_idx = get_datasets()
small_train = MothDataset(train_ds_full.samples[:SMALL_TRAIN], label_to_idx, train_ds_full.transform)
small_val   = MothDataset(val_ds_full.samples[:SMALL_VALTEST],  label_to_idx, val_ds_full.transform)
small_test  = MothDataset(test_ds_full.samples[:SMALL_VALTEST], label_to_idx, test_ds_full.transform)
print(f"Classes: {num_classes}, train={len(small_train)}, val={len(small_val)}, test={len(small_test)}")


def small_datasets(*_args, **_kwargs):
    return small_train, small_val, small_test, num_classes, label_to_idx


# Steps 3+4: train then evaluate for each model/backbone combination
RUNS = [
    ("baseline",      "resnet50"),
    ("early_fusion",  "resnet50"),
    ("late_fusion",   "resnet50"),
    ("gated_fusion",  "resnet50"),
    ("location_only", None),
]

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    for model, backbone in RUNS:
        label = f"{model}_{backbone}" if backbone else model

        print(f"\n--- train: {label} ---")
        train_argv = ["train.py", "--model", model, "--epochs", "2", "--batch_size", "4"]
        if backbone:
            train_argv += ["--backbone", backbone]

        with (
            patch.object(pipeline.train, "get_datasets", side_effect=small_datasets),
            patch.object(pipeline.train, "RESULTS_DIR", tmp),
            patch("sys.argv", train_argv),
        ):
            pipeline.train.main()

        result_dir = tmp / (f"{model}_{backbone}" if backbone else "location_only")
        assert (result_dir / "best.pt").exists(),          "best.pt missing"
        assert (result_dir / "last.pt").exists(),          "last.pt missing"
        assert (result_dir / "label_to_idx.json").exists(), "label_to_idx.json missing"

        print(f"--- evaluate: {label} ---")
        test_argv = ["test.py", "--model", model, "--batch_size", "4"]
        if backbone:
            test_argv += ["--backbone", backbone]

        with (
            patch.object(pipeline_test, "get_datasets", side_effect=small_datasets),
            patch.object(pipeline_test, "RESULTS_DIR", tmp),
            patch("sys.argv", test_argv),
        ):
            pipeline_test.main()

        assert (result_dir / "metrics.json").exists(), "metrics.json missing"
        with open(result_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert "macro_f1" in metrics
        print(f"  macro_f1 = {metrics['macro_f1']:.4f}")

print("\nAll pipeline tests passed")
