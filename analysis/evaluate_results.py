"""Load metrics.json from each results/ directory and produce comparison plots."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_DIR     = Path(__file__).parent / "output"

# Display order and labels for all 7 experiments.
EXPERIMENTS = [
    ("baseline_resnet50",    "ResNet\nbaseline"),
    ("early_fusion_resnet50","ResNet\nearly"),
    ("late_fusion_resnet50", "ResNet\nlate"),
    ("location_only",        "Location\nonly"),
    ("baseline_bioclip",     "BioCLIP\nbaseline"),
    ("early_fusion_bioclip", "BioCLIP\nearly"),
    ("late_fusion_bioclip",  "BioCLIP\nlate"),
]

BUCKETS      = ["rare", "medium", "common"]
BUCKET_LABEL = {"rare": "Rare (10–29)", "medium": "Medium (30–99)", "common": "Common (100+)"}
BUCKET_COLOR = {"rare": "#d62728", "medium": "#ff7f0e", "common": "#2ca02c"}


def load_metrics():
    """Return dict mapping experiment key → metrics dict (from metrics.json)."""
    data = {}
    for key, _ in EXPERIMENTS:
        path = RESULTS_DIR / key / "metrics.json"
        if path.exists():
            with open(path) as f:
                data[key] = json.load(f)
    return data


def plot_macro_f1(metrics):
    """Bar chart: overall macro-F1 for every available experiment."""
    keys   = [k for k, _ in EXPERIMENTS if k in metrics and "macro_f1" in metrics[k]]
    labels = [lbl for k, lbl in EXPERIMENTS if k in metrics and "macro_f1" in metrics[k]]
    values = [metrics[k]["macro_f1"] for k in keys]

    if not keys:
        print("No macro_f1 data found — skipping macro_f1_comparison.png")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#4878cf" if "resnet" in k or k == "location_only" else "#6acc65" for k in keys]
    bars = ax.bar(range(len(keys)), values, color=colors, width=0.6)

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Overall macro-F1 by experiment")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4878cf", label="ResNet-50"),
                        Patch(color="#6acc65", label="BioCLIP")],
              fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "macro_f1_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved macro_f1_comparison.png")


def plot_bucket_f1(metrics):
    """Grouped bar chart: per-bucket macro-F1 for each experiment."""
    keys   = [k for k, _ in EXPERIMENTS
              if k in metrics and "bucket_f1" in metrics[k]]
    labels = [lbl for k, lbl in EXPERIMENTS
              if k in metrics and "bucket_f1" in metrics[k]]

    if not keys:
        print("No bucket_f1 data found — skipping bucket_f1_comparison.png")
        return

    x      = np.arange(len(keys))
    width  = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(10, 4))
    for offset, bucket in zip(offsets, BUCKETS):
        values = [metrics[k]["bucket_f1"].get(bucket, float("nan")) for k in keys]
        ax.bar(x + offset, values, width, label=BUCKET_LABEL[bucket],
               color=BUCKET_COLOR[bucket], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 per species bucket by experiment")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bucket_f1_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved bucket_f1_comparison.png")


def plot_location_gain(metrics):
    """Bar chart: macro-F1 gain of fusion over baseline for each backbone."""
    pairs = [
        ("ResNet-50 early",  "baseline_resnet50", "early_fusion_resnet50"),
        ("ResNet-50 late",   "baseline_resnet50", "late_fusion_resnet50"),
        ("BioCLIP early",    "baseline_bioclip",  "early_fusion_bioclip"),
        ("BioCLIP late",     "baseline_bioclip",  "late_fusion_bioclip"),
    ]
    labels, gains = [], []
    for label, base_key, fusion_key in pairs:
        if (base_key in metrics and fusion_key in metrics
                and "macro_f1" in metrics[base_key]
                and "macro_f1" in metrics[fusion_key]):
            labels.append(label)
            gains.append(metrics[fusion_key]["macro_f1"] - metrics[base_key]["macro_f1"])

    if not gains:
        print("Not enough data for location_gain.png — skipping")
        return

    colors = ["#2ca02c" if g >= 0 else "#d62728" for g in gains]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(labels)), gains, color=colors, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ΔMacro-F1 (fusion − baseline)")
    ax.set_title("Macro-F1 gain from adding location data")

    for bar, g in zip(bars, gains):
        va  = "bottom" if g >= 0 else "top"
        off = 0.002 if g >= 0 else -0.002
        ax.text(bar.get_x() + bar.get_width() / 2, g + off, f"{g:+.3f}",
                ha="center", va=va, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "location_gain.png", dpi=150)
    plt.close(fig)
    print("Saved location_gain.png")


def load_histories():
    """Return dict mapping experiment key → list of per-epoch dicts from history.json."""
    data = {}
    for key, _ in EXPERIMENTS:
        path = RESULTS_DIR / key / "history.json"
        if path.exists():
            with open(path) as f:
                data[key] = json.load(f)
    return data


def plot_learning_curves(histories):
    """Train/val loss curves for all available experiments, one subplot each."""
    keys   = [k for k, _ in EXPERIMENTS if k in histories]
    labels = {k: lbl for k, lbl in EXPERIMENTS}

    if not keys:
        print("No history.json files found — skipping learning_curves.png")
        return

    ncols = min(4, len(keys))
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for ax, key in zip(axes.flat, keys):
        h      = histories[key]
        epochs = [e["epoch"] for e in h]
        ax.plot(epochs, [e["train_loss"] for e in h], label="train", color="#4878cf")
        ax.plot(epochs, [e["val_loss"]   for e in h], label="val",   color="#d62728")
        ax.set_title(labels[key].replace("\n", " "), fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)

    fig.suptitle("Training and validation loss curves", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "learning_curves.png", dpi=150)
    plt.close(fig)
    print("Saved learning_curves.png")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    metrics   = load_metrics()
    histories = load_histories()

    if not metrics and not histories:
        print(f"No results found in {RESULTS_DIR}/ — run ALICE jobs first.")
    else:
        available = set(metrics) | set(histories)
        print(f"Loaded results for: {', '.join(available)}")

    plot_macro_f1(metrics)
    plot_bucket_f1(metrics)
    plot_location_gain(metrics)
    plot_learning_curves(histories)
