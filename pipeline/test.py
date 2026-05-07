#!/usr/bin/env python3
"""Evaluate a trained model on the test set and optionally use Grad-CAM."""
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

METADATA_ONLY_MODELS = {"location_only", "metadata_only"}


def get_result_dir(model_name, backbone):
    if model_name in METADATA_ONLY_MODELS:
        return RESULTS_DIR / model_name
    return RESULTS_DIR / f"{model_name}_{backbone}"


def build_model(model_name, backbone, num_classes, metadata_dim):
    if model_name == "baseline":
        from models.baseline import BaselineModel
        return BaselineModel(num_classes, backbone)
    if model_name == "location_only":
        from models.location_only import LocationOnlyModel
        return LocationOnlyModel(num_classes, metadata_dim)
    if model_name == "metadata_only":
        from models.metadata_only import MetadataOnlyModel
        return MetadataOnlyModel(num_classes, metadata_dim)
    if model_name == "early_fusion":
        from models.early_fusion import EarlyFusionModel
        return EarlyFusionModel(num_classes, backbone, metadata_dim)
    if model_name == "late_fusion":
        from models.late_fusion import LateFusionModel
        return LateFusionModel(num_classes, backbone, metadata_dim)
    if model_name == "gated_fusion":
        from models.gated_fusion import GatedFusionModel
        return GatedFusionModel(num_classes, backbone, metadata_dim)
    if model_name == "concat_fusion":
        from models.concat_fusion import ConcatFusionModel
        return ConcatFusionModel(num_classes, metadata_dim, backbone)
    if model_name == "transformer_fusion":
        from models.transformer_fusion import TransformerFusionModel
        return TransformerFusionModel(num_classes, metadata_dim, backbone)
    if model_name == "coordination":
        from models.coordination import CoordinationModel
        return CoordinationModel(num_classes, metadata_dim, backbone)
    raise ValueError(f"Unknown model: {model_name}")


def get_species_counts(train_ds):
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
    model.eval()
    all_preds, all_labels = [], []
    n        = len(loader)
    interval = max(1, n // 100)
    for i, (img, meta, label) in enumerate(loader):
        img, meta = img.to(device), meta.to(device)
        out = model(img, meta)
        logits = out[0] if isinstance(out, tuple) else out
        all_preds.extend(logits.argmax(1).cpu().tolist())
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


# ---- plots ----

def save_confusion_matrix(preds, labels, idx_to_species, result_dir):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    plots_dir = result_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    num_classes = len(idx_to_species)
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(max(12, num_classes // 6),
                                    max(10, num_classes // 6)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=12)
    ax.tick_params(axis="both", labelsize=5)
    fig.tight_layout()
    path = plots_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def save_per_species_accuracy(preds, labels, idx_to_species, result_dir):
    import matplotlib.pyplot as plt

    plots_dir = result_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    num_classes = len(idx_to_species)
    correct = np.zeros(num_classes)
    total   = np.zeros(num_classes)
    for p, l in zip(preds, labels):
        total[l]   += 1
        correct[l] += int(p == l)

    accs    = np.where(total > 0, correct / total, 0.0)
    species = [idx_to_species[i] for i in range(num_classes)]
    order   = np.argsort(accs)

    fig, ax = plt.subplots(figsize=(10, max(6, num_classes * 0.18)))
    ax.barh(range(num_classes), accs[order], color="#4878cf", height=0.7)
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels([species[i] for i in order], fontsize=5)
    ax.set_xlabel("Accuracy", fontsize=10)
    ax.set_title("Per-species accuracy (sorted)", fontsize=12)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    path = plots_dir / "per_species_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---- Grad-CAM ----

def vit_reshape_transform(tensor):
    if tensor.shape[0] == 197:
        tensor = tensor.permute(1, 0, 2)
    patch_tokens = tensor[:, 1:, :]
    patch_tokens = patch_tokens.reshape(
        patch_tokens.shape[0], 14, 14, patch_tokens.shape[2])
    return patch_tokens.permute(0, 3, 1, 2)


def run_gradcam(model, test_ds, backbone, result_dir, preds_all, species_counts, device):
    from pytorch_grad_cam import HiResCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from PIL import Image as PILImage

    if backbone == "resnet50":
        cam = HiResCAM(model=model, target_layers=[model.backbone.layer4[-1]])
    else:
        cam = HiResCAM(model=model,
                       target_layers=[model.backbone.transformer.resblocks[-1].ln_1],
                       reshape_transform=vit_reshape_transform)

    rng = random.Random(GRADCAM_SEED)
    bucket_correct = defaultdict(list)
    bucket_wrong   = defaultdict(list)

    for idx in range(len(test_ds)):
        _, _, label = test_ds[idx]
        label  = int(label)
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
        img_t, meta_t, true_label = test_ds[idx]
        img_batch  = img_t.unsqueeze(0).to(device)
        meta_batch = meta_t.unsqueeze(0).to(device)

        targets = [ClassifierOutputTarget(int(true_label))] if outcome == "wrong" else None
        grayscale_cam = cam(input_tensor=img_batch, targets=targets)[0]

        rgb     = (img_t * inv_std + inv_mean).permute(1, 2, 0).numpy().clip(0, 1)
        overlay = show_cam_on_image(rgb.astype(np.float32), grayscale_cam, use_rgb=True)
        PILImage.fromarray(overlay).save(gradcam_dir / f"{bucket}_{outcome}_{idx}.png")

    print(f"Grad-CAM: {len(selected)} images saved to {gradcam_dir}/")


# ---- Main ----

def main():
    all_models = [
        "baseline", "location_only", "metadata_only",
        "early_fusion", "late_fusion", "gated_fusion",
        "concat_fusion", "transformer_fusion", "coordination",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True, choices=all_models)
    parser.add_argument("--backbone",   default="resnet50", choices=["resnet50", "bioclip"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradcam",    action="store_true",
                        help="Enable Grad-CAM visualisation for image models")
    parser.add_argument("--plots",      action="store_true",
                        help="Save confusion matrix and per-species accuracy chart")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = get_result_dir(args.model, args.backbone)

    best_pt = result_dir / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"{best_pt} not found. Training output is required.")

    label_to_idx_path = result_dir / "label_to_idx.json"
    if not label_to_idx_path.exists():
        raise FileNotFoundError(f"{label_to_idx_path} not found. label_to_idx.json is required.")
    with open(label_to_idx_path) as f:
        label_to_idx = json.load(f)
    num_classes   = len(label_to_idx)
    idx_to_species = {v: k for k, v in label_to_idx.items()}

    print("Loading datasets...")
    train_ds, _, test_ds, _, _, metadata_dim = get_datasets(backbone=args.backbone)
    species_counts = get_species_counts(train_ds)

    print("Building model and loading best.pt...")
    model = build_model(args.model, args.backbone, num_classes, metadata_dim).to(device)
    ckpt  = torch.load(best_pt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=8, pin_memory=True,
                             persistent_workers=False, prefetch_factor=2)

    print("Running inference on test set...")
    preds, labels = run_inference(model, test_loader, device)

    macro_f1, bucket_f1 = compute_metrics(preds, labels, species_counts)
    acc = float((preds == labels).mean())
    print(f"Accuracy:           {acc:.4f}")
    print(f"Macro-F1 (overall): {macro_f1:.4f}")
    for bucket, score in bucket_f1.items():
        print(f"  {bucket:6s}: {score:.4f}" if score is not None else f"  {bucket:6s}: n/a")

    metrics = {"accuracy": acc, "macro_f1": macro_f1, "bucket_f1": bucket_f1}
    metrics_path = result_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    if args.plots:
        print("Saving plots...")
        save_confusion_matrix(preds, labels, idx_to_species, result_dir)
        save_per_species_accuracy(preds, labels, idx_to_species, result_dir)

    if args.gradcam:
        if args.model in METADATA_ONLY_MODELS:
            print("Grad-CAM is not available for metadata-only models, skipping.")
        else:
            run_gradcam(model, test_ds, args.backbone, result_dir,
                        preds, species_counts, device)


if __name__ == "__main__":
    main()
