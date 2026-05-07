"""Fission analysis for the pipeline.

Reads metrics.json from each result directory and runs the missing-modality
test on the best fusion model.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

RESULTS_DIR  = Path(__file__).parent.parent / "results"
OUT_DIR      = Path(__file__).parent / "output"
METADATA_ONLY_MODELS = {"location_only", "metadata_only"}

# All experiments in display order
EXPERIMENTS = [
    ("baseline",          "resnet50", "Baseline ResNet-50"),
    ("baseline",          "bioclip",  "Baseline BioCLIP"),
    ("location_only",     None,       "Location Only"),
    ("metadata_only",     None,       "Metadata Only"),
    ("early_fusion",      "resnet50", "Early Fusion ResNet-50"),
    ("early_fusion",      "bioclip",  "Early Fusion BioCLIP"),
    ("late_fusion",       "resnet50", "Late Fusion ResNet-50"),
    ("late_fusion",       "bioclip",  "Late Fusion BioCLIP"),
    ("gated_fusion",      "resnet50", "Gated Fusion ResNet-50"),
    ("gated_fusion",      "bioclip",  "Gated Fusion BioCLIP"),
    ("concat_fusion",     "resnet50", "Concat Fusion ResNet-50"),
    ("concat_fusion",     "bioclip",  "Concat Fusion BioCLIP"),
    ("transformer_fusion","resnet50", "Transformer Fusion ResNet-50"),
    ("transformer_fusion","bioclip",  "Transformer Fusion BioCLIP"),
    ("coordination",      "resnet50", "Coordination ResNet-50"),
    ("coordination",      "bioclip",  "Coordination BioCLIP"),
]

FUSION_MODELS = {
    "early_fusion", "late_fusion", "gated_fusion",
    "concat_fusion", "transformer_fusion", "coordination",
}

UNIMODAL_MODELS = {"baseline", "location_only", "metadata_only"}


def result_dir(model_name, backbone):
    if model_name in METADATA_ONLY_MODELS:
        return RESULTS_DIR / model_name
    return RESULTS_DIR / f"{model_name}_{backbone}"


def load_all_metrics():
    metrics = {}
    for model, backbone, label in EXPERIMENTS:
        path = result_dir(model, backbone) / "metrics.json"
        if path.exists():
            with open(path) as f:
                metrics[(model, backbone)] = json.load(f)
    return metrics


# ---- Missing modality test ----

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


@torch.no_grad()
def run_missing_modality(model, loader, device, zero_out):
    """Evaluate model with one modality zeroed out. Returns accuracy."""
    model.eval()
    correct, total = 0, 0
    for img, meta, label in loader:
        img, meta, label = img.to(device), meta.to(device), label.to(device)
        if zero_out == "metadata":
            meta = torch.zeros_like(meta)
        elif zero_out == "image":
            img = torch.zeros_like(img)
        out    = model(img, meta)
        logits = out[0] if isinstance(out, tuple) else out
        correct += (logits.argmax(1) == label).sum().item()
        total   += label.size(0)
    return correct / total * 100


def missing_modality_test(best_model_name, best_backbone, device):
    from pipeline.preprocessing import get_datasets
    print("\nLoading datasets for missing modality test...")
    train_ds, _, test_ds, num_classes, label_to_idx, metadata_dim = get_datasets(
        backbone=best_backbone)

    rdir     = result_dir(best_model_name, best_backbone)
    best_pt  = rdir / "best.pt"
    if not best_pt.exists():
        print(f"  Checkpoint not found: {best_pt}. Missing-modality test skipped.")
        return None, None

    model = build_model(best_model_name, best_backbone, num_classes, metadata_dim).to(device)
    ckpt  = torch.load(best_pt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)

    acc_no_meta  = run_missing_modality(model, loader, device, zero_out="metadata")
    acc_no_image = run_missing_modality(model, loader, device, zero_out="image")
    return acc_no_meta, acc_no_image


# ---- Output helpers ----

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files: f.flush()


def _sep(n=60): print("=" * n)


# ---- Main ----

def main():
    OUT_DIR.mkdir(exist_ok=True)
    log_file   = open(OUT_DIR / "log_fission_analysis.txt", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = load_all_metrics()

    if not metrics:
        print(f"No metrics.json files found in {RESULTS_DIR}/")
        print("Metrics from pipeline/test.py are required.")
        return

    # ---- Step 1: Full results table ----
    _sep()
    print("STEP 1: MODEL RESULTS")
    _sep()
    print(f"\n{'Model':<35} {'Macro-F1':>10} {'Accuracy':>10}")
    print("-" * 57)

    results = {}
    for model, backbone, label in EXPERIMENTS:
        key = (model, backbone)
        if key not in metrics:
            continue
        m       = metrics[key]
        f1      = m.get("macro_f1", float("nan"))
        acc     = m.get("accuracy", float("nan"))
        results[key] = {"macro_f1": f1, "accuracy": acc, "label": label,
                        "model": model, "backbone": backbone}
        print(f"{label:<35} {f1:>10.4f} {acc:>9.1%}")

    # ---- Step 2: Fission analysis ----
    _sep()
    print("\nSTEP 2: FISSION ANALYSIS")
    _sep()

    unimodal_f1 = {k: v["macro_f1"] for k, v in results.items()
                   if v["model"] in UNIMODAL_MODELS}
    fusion_f1   = {k: v["macro_f1"] for k, v in results.items()
                   if v["model"] in FUSION_MODELS}

    if not unimodal_f1 or not fusion_f1:
        print("Not enough models evaluated for fission analysis.")
        return

    best_uni_key   = max(unimodal_f1, key=unimodal_f1.get)
    best_fusion_key = max(fusion_f1,  key=fusion_f1.get)

    best_uni_f1    = unimodal_f1[best_uni_key]
    best_fusion_f1 = fusion_f1[best_fusion_key]

    # Image-only and metadata-only baselines
    img_f1  = max((v["macro_f1"] for k, v in results.items()
                   if v["model"] == "baseline"), default=float("nan"))
    meta_f1 = max((v["macro_f1"] for k, v in results.items()
                   if v["model"] in {"location_only", "metadata_only"}),
                  default=float("nan"))

    print(f"\nBest image-only model:    {img_f1:.4f} macro-F1")
    print(f"Best metadata-only model: {meta_f1:.4f} macro-F1")
    print(f"Best fusion model:        {results[best_fusion_key]['label']}")
    print(f"Best fusion macro-F1:     {best_fusion_f1:.4f}")

    synergy = best_fusion_f1 - best_uni_f1
    print(f"\nSynergy (fusion - best unimodal): {synergy:+.4f}")
    if synergy > 0:
        print("-> Fusion captures SYNERGISTIC information")
    else:
        print("-> No clear synergy detected")

    # Coordination benefit
    coord_f1s = {k: v["macro_f1"] for k, v in results.items()
                 if v["model"] == "coordination"}
    if coord_f1s:
        best_coord_f1  = max(coord_f1s.values())
        coord_benefit  = best_coord_f1 - best_fusion_f1
        print(f"\nBest coordination macro-F1: {best_coord_f1:.4f}")
        print(f"Coordination benefit:       {coord_benefit:+.4f}")
        if coord_benefit > 0:
            print("-> Contrastive alignment IMPROVES fusion")
        else:
            print("-> Coordination did not help over simple fusion")

    # ---- Step 3: Missing modality test ----
    _sep()
    print("\nSTEP 3: MISSING MODALITY TEST")
    _sep()

    bm      = results[best_fusion_key]
    full_acc = bm["accuracy"] * 100

    print(f"\nBest fusion model: {bm['label']}")
    print(f"Full modalities:   {full_acc:.2f}%")

    acc_no_meta, acc_no_image = missing_modality_test(
        bm["model"], bm["backbone"], device)

    if acc_no_meta is not None:
        print(f"Missing metadata:  {acc_no_meta:.2f}%  "
              f"(drop: {full_acc - acc_no_meta:+.2f}%)")
        print(f"Missing image:     {acc_no_image:.2f}%  "
              f"(drop: {full_acc - acc_no_image:+.2f}%)")

        if full_acc - acc_no_meta > full_acc - acc_no_image:
            print("\n-> Metadata contributes more than image to this fusion model")
        else:
            print("\n-> Image contributes more than metadata to this fusion model")

    # ---- Step 4: BioCLIP vs ResNet-50 gain ----
    _sep()
    print("\nSTEP 4: BioCLIP vs ResNet-50 GAIN")
    _sep()
    print(f"\n{'Model':<25} {'ResNet-50':>10} {'BioCLIP':>10} {'Gain':>8}")
    print("-" * 55)
    for model in ["baseline", "early_fusion", "late_fusion", "gated_fusion",
                  "concat_fusion", "transformer_fusion", "coordination"]:
        r50 = results.get((model, "resnet50"), {}).get("macro_f1")
        bio = results.get((model, "bioclip"),  {}).get("macro_f1")
        if r50 is not None and bio is not None:
            print(f"{model:<25} {r50:>10.4f} {bio:>10.4f} {bio - r50:>+8.4f}")

    log_file.close()
    sys.stdout = sys.__stdout__
    print(f"\nFission analysis saved to {OUT_DIR}/log_fission_analysis.txt")


if __name__ == "__main__":
    main()
