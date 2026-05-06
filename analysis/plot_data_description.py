"""Generate data description plots for the paper."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OCCURRENCE = Path(__file__).parent.parent / "data" / "occurrence.txt"
OUT_DIR    = Path(__file__).parent / "output"  # saves plots to analysis/output
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_data():
    counts        = Counter()
    month_records = defaultdict(int)
    month_species = defaultdict(set)

    with open(OCCURRENCE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sp = row.get("scientificName", "").strip()
            m  = row.get("month", "").strip()
            if not (sp and m):
                continue
            counts[sp] += 1
            month_records[int(m)] += 1
            month_species[int(m)].add(sp)

    return counts, month_records, month_species


def plot_observations_per_species(counts):
    fig, ax = plt.subplots(figsize=(8, 4))

    values = sorted(counts.values(), reverse=True)
    ax.bar(range(len(values)), values, width=1.0, color="steelblue", linewidth=0)
    ax.set_xlabel("Species (ranked by observations)")
    ax.set_ylabel("Number of observations")
    ax.set_title("Observations per species (all 745 species)")
    ax.set_yscale("log")

    for threshold, color in [(100, "#d62728"), (30, "#ff7f0e"), (10, "#2ca02c")]:
        n_kept = sum(1 for v in values if v >= threshold)
        ax.axvline(n_kept, color=color, linestyle="--", linewidth=1.2,
                   label=f"≥{threshold} samples ({n_kept} species)")
        ax.axhline(threshold, color=color, linestyle=":", linewidth=0.8, alpha=0.6)

    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "obs_per_species.png", dpi=150)
    plt.close(fig)
    print("Saved obs_per_species.png")


def plot_monthly_distribution(month_records, month_species):
    months   = list(range(1, 13))
    records  = [month_records[m] for m in months]
    n_species = [len(month_species[m]) for m in months]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    x = np.arange(12)
    bars = ax1.bar(x, records, color="steelblue", alpha=0.7, label="Observations")
    ax2.plot(x, n_species, color="#2ca02c", marker="o", linewidth=2,
             markersize=5, label="Unique species")

    ax1.set_xticks(x)
    ax1.set_xticklabels(MONTH_NAMES)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Number of observations", color="steelblue")
    ax2.set_ylabel("Number of unique species", color="#2ca02c")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax1.set_title("Seasonal distribution of observations and species")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "monthly_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved monthly_distribution.png")


if __name__ == "__main__":
    counts, month_records, month_species = load_data()
    plot_observations_per_species(counts)
    plot_monthly_distribution(month_records, month_species)
