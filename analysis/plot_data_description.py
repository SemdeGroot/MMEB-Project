"""Generate data description plots for the paper."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OCCURRENCE = Path(__file__).parent.parent / "data" / "occurrence.txt"
OUT_DIR    = Path(__file__).parent / "output"  # saves plots to analysis/output
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Approximate Netherlands bounding box (matches docs/assumptions.txt item 2).
NL_LAT = (50.7, 53.6)
NL_LON = (3.3, 7.2)


def load_data():
    counts        = Counter()
    month_records = defaultdict(int)
    month_species = defaultdict(set)
    coords        = []  # (lat, lon) per record with valid coordinates

    with open(OCCURRENCE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sp = row.get("scientificName", "").strip()
            m  = row.get("month", "").strip()
            if not (sp and m):
                continue
            counts[sp] += 1
            month_records[int(m)] += 1
            month_species[int(m)].add(sp)

            lat = row.get("decimalLatitude", "").strip()
            lon = row.get("decimalLongitude", "").strip()
            if lat and lon:
                try:
                    coords.append((float(lat), float(lon)))
                except ValueError:
                    pass

    return counts, month_records, month_species, coords


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


def plot_geographic_distribution(coords):
    """Hexbin density of observations on a map of the Netherlands."""
    lats = np.array([c[0] for c in coords])
    lons = np.array([c[1] for c in coords])

    in_box = ((lats >= NL_LAT[0]) & (lats <= NL_LAT[1])
              & (lons >= NL_LON[0]) & (lons <= NL_LON[1]))
    n_out = int((~in_box).sum())
    lats, lons = lats[in_box], lons[in_box]

    # Tight extent: just a sliver of margin so the outermost hexbins aren't clipped.
    pad = 0.05
    extent = [lons.min() - pad, lons.max() + pad,
              lats.min() - pad, lats.max() + pad]

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={"projection": proj})
    ax.set_extent(extent, crs=proj)

    # 10 m NaturalEarth features give recognisable NL coastline and borders.
    ax.add_feature(cfeature.OCEAN.with_scale("10m"),     facecolor="#e8eef4")
    ax.add_feature(cfeature.LAND.with_scale("10m"),      facecolor="#f5f3ec")
    ax.add_feature(cfeature.LAKES.with_scale("10m"),     facecolor="#e8eef4", linewidth=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"),   linewidth=0.5, linestyle=":")

    # Drop the lightest 30% of YlOrRd so even sparse hexbins read clearly on the land.
    base   = plt.colormaps["YlOrRd"]
    cmap   = LinearSegmentedColormap.from_list("YlOrRd_dark", base(np.linspace(0.30, 1.0, 256)))

    hb = ax.hexbin(lons, lats, gridsize=45, cmap=cmap,
                   mincnt=1, bins="log", transform=proj)
    cb = fig.colorbar(hb, ax=ax, shrink=0.75, pad=0.04)
    cb.set_label("Observations per hexbin (log scale)")

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(f"Geographic distribution of observations (n={len(lats):,})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "geographic_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved geographic_distribution.png ({n_out} records outside NL bbox dropped)")


if __name__ == "__main__":
    counts, month_records, month_species, coords = load_data()
    plot_observations_per_species(counts)
    plot_monthly_distribution(month_records, month_species)
    plot_geographic_distribution(coords)
