#!/usr/bin/env python3
"""Evaluate a trained model on the test set and optionally run Grad-CAM."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from .preprocessing import get_datasets

RESULTS_DIR = Path(__file__).parent.parent / "results"

BUCKET_RANGES = {
    "rare":   (10,  29),
    "medium": (30,  99),
    "common": (100, float("inf")),
}

GRADCAM_PER_BUCKET = 3
GRADCAM_SEED       = 42


def get_result_dir(model_name, backbone):
    if model_name == "location_only":
        return RESULTS_DIR / "location_only"
    return RESULTS_DIR / f"{model_name}_{backbone}"


def build_model(model_name, backbone, num_classes):
    if model_name == "baseline":
        from models.baseline import BaselineModel
        return BaselineModel(num_classes, backbone)
    if model_name == "early_fusion":
        from models.early_fusion import EarlyFusionModel
        return EarlyFusionModel(num_classes, backbone)
    if model_name == "late_fusion":
        from models.late_fusion import LateFusionModel
        return LateFusionModel(num_classes, backbone)
    if model_name == "gated_fusion":
        from models.gated_fusion import GatedFusionModel
        return GatedFusionModel(num_classes, backbone)
    if model_name == "location_only":
        from models.location_only import LocationOnlyModel
        return LocationOnlyModel(num_classes)
    raise ValueError(f"Unknown model: {model_name}")


def get_species_counts(train_ds):
    """Return dict mapping class index → number of unique training gbifIDs.

    Counted at the occurrence (gbifID) level, not the image level: bucket
    thresholds (rare 10–29, medium 30–99, common 100+) are defined as records,
    not photos, so multiple photos of one observation must not inflate the count.
    """
    gbifids_per_class = defaultdict(set)
    for sample in train_ds.samples:
        cls = train_ds.label_to_idx[sample[-1]]
        gbifids_per_class[cls].add(sample[1])
    return {cls: len(ids) for cls, ids in gbifids_per_class.items()}


def bucket_for_count(count):
    for name, (lo, hi) in BUCKET_RANGES.items():
        if lo <= count <= hi:
            return name
    return None


def _progress(tag, i, n):
    filled = int(40 * (i + 1) / n)
    bar = "#" * filled + "-" * (40 - filled)
    print(f"  {tag} [{bar}] {i+1}/{n}", flush=True)


@torch.no_grad()
def run_inference(model, loader, device):
    """Return (predictions, labels) as numpy arrays over the full loader."""
    model.eval()
    all_preds, all_labels = [], []
    n = len(loader)
    interval = max(1, n // 100)
    for i, (img, loc, label) in enumerate(loader):
        img, loc = img.to(device), loc.to(device)
        all_preds.extend(model(img, loc).argmax(1).cpu().tolist())
        all_labels.extend(label.tolist())
        if (i + 1) % interval == 0 or i + 1 == n:
            _progress("inference", i, n)
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels, species_counts):
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))

    bucket_f1 = {}
    for name, (lo, hi) in BUCKET_RANGES.items():
        bucket_classes = [cls for cls, cnt in species_counts.items() if lo <= cnt <= hi]
        mask = np.isin(labels, bucket_classes)
        if mask.sum() == 0:
            bucket_f1[name] = None
            continue
        bucket_f1[name] = float(f1_score(
            labels[mask], preds[mask],
            labels=bucket_classes, average="macro", zero_division=0,
        ))

    return macro_f1, bucket_f1


def vit_reshape_transform(tensor):
    # open_clip's ViT may emit tensors in either (B, 197, D) or (197, B, D);
    # 197 = 14*14 patches + 1 CLS. Detect by which axis is 197 and permute to
    # batch-first before dropping CLS and reshaping to a 14x14 spatial grid.
    if tensor.shape[0] == 197:
        tensor = tensor.permute(1, 0, 2)
    patch_tokens = tensor[:, 1:, :]
    patch_tokens = patch_tokens.reshape(patch_tokens.shape[0], 14, 14, patch_tokens.shape[2])
    return patch_tokens.permute(0, 3, 1, 2)


def run_gradcam(model, test_ds, backbone, result_dir, preds_all, species_counts, device):
    from pytorch_grad_cam import HiResCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from PIL import Image as PILImage

    # HiResCAM for both backbones: necessary for ViT (vanilla GradCAM zeros out
    # the patch-token map after ReLU and produces uniform blue images), and
    # also strictly more faithful for CNNs per Draelos & Carin (2020). One
    # method across both keeps the saliency comparison consistent.
    if backbone == "resnet50":
        cam = HiResCAM(model=model, target_layers=[model.backbone.layer4[-1]])
    else:
        # ln_1 (LayerNorm before attention in the last block) matches the
        # official pytorch_grad_cam ViT example. Hooking on the resblock output
        # picks up the residual stream and smears the CAM.
        cam = HiResCAM(model=model,
                       target_layers=[model.backbone.transformer.resblocks[-1].ln_1],
                       reshape_transform=vit_reshape_transform)

    # Group test indices by (bucket, correct/incorrect)
    rng = random.Random(GRADCAM_SEED)
    bucket_correct = defaultdict(list)
    bucket_wrong   = defaultdict(list)

    for idx in range(len(test_ds)):
        _, _, label = test_ds[idx]
        label = int(label)
        cnt    = species_counts.get(label, 0)
        bucket = bucket_for_count(cnt)
        if bucket is None:
            continue
        if preds_all[idx] == label:
            bucket_correct[bucket].append(idx)
        else:
            bucket_wrong[bucket].append(idx)

    selected = []
    for bucket in ["rare", "medium", "common"]:
        for group, outcome in [(bucket_correct[bucket], "correct"),
                               (bucket_wrong[bucket],   "wrong")]:
            chosen = rng.sample(group, min(GRADCAM_PER_BUCKET, len(group)))
            selected.extend((idx, bucket, outcome) for idx in chosen)

    gradcam_dir = result_dir / "gradcam"
    gradcam_dir.mkdir(exist_ok=True)

    inv_mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    inv_std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    model.eval()
    for idx, bucket, outcome in selected:
        img_t, loc_t, true_label = test_ds[idx]
        img_batch = img_t.unsqueeze(0).to(device)
        loc_batch = loc_t.unsqueeze(0).to(device)

        # For wrong predictions, target the true class so the heatmap shows where the model
        # failed to look. For correct predictions, argmax (None) is the true class anyway.
        if outcome == "wrong":
            targets = [ClassifierOutputTarget(int(true_label))]
        else:
            targets = None

        grayscale_cam = cam(input_tensor=img_batch, targets=targets)[0]

        rgb = (img_t * inv_std + inv_mean).permute(1, 2, 0).numpy().clip(0, 1)
        overlay = show_cam_on_image(rgb.astype(np.float32), grayscale_cam, use_rgb=True)
        PILImage.fromarray(overlay).save(gradcam_dir / f"{bucket}_{outcome}_{idx}.png")

    print(f"Grad-CAM: {len(selected)} images saved to {gradcam_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True,
                        choices=["baseline", "early_fusion", "late_fusion", "gated_fusion", "location_only"])
    parser.add_argument("--backbone",   default="resnet50", choices=["resnet50", "bioclip"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradcam",    action="store_true",
                        help="Run Grad-CAM visualisation (baseline models only)")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = get_result_dir(args.model, args.backbone)

    best_pt = result_dir / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"{best_pt} not found — run train.py first")

    label_to_idx_path = result_dir / "label_to_idx.json"
    if not label_to_idx_path.exists():
        raise FileNotFoundError(f"{label_to_idx_path} not found — run train.py first")
    with open(label_to_idx_path) as f:
        label_to_idx = json.load(f)
    num_classes = len(label_to_idx)

    print("Loading datasets...")
    train_ds, _, test_ds, _, _ = get_datasets(backbone=args.backbone)
    species_counts = get_species_counts(train_ds)

    print("Building model and loading best.pt...")
    model = build_model(args.model, args.backbone, num_classes).to(device)
    ckpt  = torch.load(best_pt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=8, pin_memory=True,
                             persistent_workers=False, prefetch_factor=2)

    print("Running inference on test set...")
    preds, labels = run_inference(model, test_loader, device)

    macro_f1, bucket_f1 = compute_metrics(preds, labels, species_counts)
    print(f"Macro-F1 (overall): {macro_f1:.4f}")
    for bucket, score in bucket_f1.items():
        print(f"  {bucket:6s}: {score:.4f}" if score is not None else f"  {bucket:6s}: n/a")

    metrics = {"macro_f1": macro_f1, "bucket_f1": bucket_f1}
    metrics_path = result_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    if args.gradcam:
        if args.model != "baseline":
            print("Grad-CAM is only implemented for baseline models, skipping.")
        else:
            run_gradcam(model, test_ds, args.backbone, result_dir, preds, species_counts, device)


if __name__ == "__main__":
    main()
