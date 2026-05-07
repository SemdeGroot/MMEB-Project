"""Exploratory data analysis with cleaning statistics and distribution plots.

Command example:
    python -m analysis.explore_data
Outputs are saved to results/explore_data/.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---- Paths aligned with pipeline/preprocessing.py ----
DATA_DIR    = Path(__file__).parent.parent / "data"
IMAGES_DIR  = DATA_DIR / "images"
OCCURRENCE  = DATA_DIR / "occurrence.txt"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "explore_data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MIN_GBIFIDS = 10

STATE_PROVINCE_NORMALIZATION = {
    'Frisia':        'Friesland',
    'North Brabant': 'Noord-Brabant',
    'North Holland': 'Noord-Holland',
    'South Holland': 'Zuid-Holland',
}

MONTH_NAMES = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
               7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}


# ---- Tee: mirror stdout to log file ----
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def load_occurrence():
    """Return list of dicts, one per row in occurrence.txt that has required fields."""
    rows = []
    with open(OCCURRENCE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gid     = row.get("gbifID", "").strip()
            species = row.get("scientificName", "").strip()
            lat     = row.get("decimalLatitude", "").strip()
            lon     = row.get("decimalLongitude", "").strip()
            month   = row.get("month", "").strip()
            day     = row.get("day", "").strip()
            prov    = row.get("stateProvince", "").strip()
            year    = row.get("year", "").strip()
            rows.append({
                "gbifID":  gid,
                "species": species,
                "lat":     lat,
                "lon":     lon,
                "month":   month,
                "day":     day,
                "province": prov,
                "year":    year,
            })
    return rows


def normalize_province(s):
    p = (s or "").strip() or "Unknown"
    return STATE_PROVINCE_NORMALIZATION.get(p, p)


def main():
    log_path = RESULTS_DIR / "log_explore_data.txt"
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print("=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)

    # ---- Step 1: raw rows ----
    print("\nLoading occurrence.txt ...")
    all_rows = load_occurrence()
    print(f"Step 1 - Raw rows in occurrence.txt:        {len(all_rows)}")

    # ---- Step 2: rows with required fields ----
    has_fields = [r for r in all_rows
                  if r["gbifID"] and r["species"] and r["lat"]
                  and r["lon"] and r["month"] and r["day"]]
    dropped_fields = len(all_rows) - len(has_fields)
    print(f"Step 2 - After dropping missing fields:      {len(has_fields)}  "
          f"(-{dropped_fields})")

    # ---- Step 3: rows that have a downloaded image ----
    image_gbifids = set()
    for p in IMAGES_DIR.glob("*/*.jpg"):
        image_gbifids.add(p.stem.split("_")[0])

    has_image = [r for r in has_fields if r["gbifID"] in image_gbifids]
    dropped_noimg = len(has_fields) - len(has_image)
    print(f"Step 3 - After keeping only downloaded imgs: {len(has_image)}  "
          f"(-{dropped_noimg})")

    # ---- Step 4: apply MIN_GBIFIDS filter ----
    species_gbifids = defaultdict(set)
    for r in has_image:
        species_gbifids[r["species"]].add(r["gbifID"])

    allowed_species = {sp for sp, ids in species_gbifids.items()
                       if len(ids) >= MIN_GBIFIDS}
    final = [r for r in has_image if r["species"] in allowed_species]
    dropped_rare   = len(species_gbifids) - len(allowed_species)
    dropped_records = len(has_image) - len(final)
    print(f"Step 4 - After MIN_GBIFIDS >= {MIN_GBIFIDS}:           {len(final)}  "
          f"(-{dropped_records} records, -{dropped_rare} species)")

    # ---- Basic stats ----
    print("\n" + "=" * 60)
    print("BASIC STATISTICS (after filtering)")
    print("=" * 60)
    print(f"Total samples:    {len(final)}")
    print(f"Total species:    {len(allowed_species)}")

    years = [int(r["year"]) for r in final if r["year"].isdigit()]
    if years:
        print(f"Year range:       {min(years)} - {max(years)}")

    lats = [float(r["lat"]) for r in final]
    lons = [float(r["lon"]) for r in final]
    print(f"Latitude range:   {min(lats):.2f} - {max(lats):.2f}")
    print(f"Longitude range:  {min(lons):.2f} - {max(lons):.2f}")

    # ---- Species distribution ----
    print("\n" + "=" * 60)
    print("SPECIES DISTRIBUTION")
    print("=" * 60)
    sp_counts = defaultdict(int)
    sp_gbifids = defaultdict(set)
    for r in final:
        sp_counts[r["species"]] += 1
        sp_gbifids[r["species"]].add(r["gbifID"])

    counts = sorted(sp_counts.values(), reverse=True)
    print(f"Samples per species  mean: {np.mean(counts):.1f}")
    print(f"Samples per species  std:  {np.std(counts):.1f}")
    print(f"Samples per species  min:  {min(counts)}")
    print(f"Samples per species  max:  {max(counts)}")

    gbif_counts = sorted(len(v) for v in sp_gbifids.values())
    print(f"gbifIDs per species  mean: {np.mean(gbif_counts):.1f}")
    print(f"gbifIDs per species  min:  {min(gbif_counts)}")
    print(f"gbifIDs per species  max:  {max(gbif_counts)}")

    top10 = sorted(sp_counts.items(), key=lambda x: -x[1])[:10]
    print("\nTop 10 most common species:")
    for sp, c in top10:
        print(f"  {c:6d}  {sp}")
    bot10 = sorted(sp_counts.items(), key=lambda x: x[1])[:10]
    print("\nTop 10 rarest species (after filter):")
    for sp, c in bot10:
        print(f"  {c:6d}  {sp}")

    # ---- Month distribution ----
    print("\n" + "=" * 60)
    print("MONTH DISTRIBUTION")
    print("=" * 60)
    month_counts = defaultdict(int)
    for r in final:
        try:
            month_counts[int(r["month"])] += 1
        except ValueError:
            pass
    for m in range(1, 13):
        print(f"  {MONTH_NAMES[m]}: {month_counts.get(m, 0)}")

    # ---- Province distribution ----
    print("\n" + "=" * 60)
    print("PROVINCE DISTRIBUTION")
    print("=" * 60)
    prov_counts = defaultdict(int)
    for r in final:
        prov_counts[normalize_province(r["province"])] += 1
    for prov, cnt in sorted(prov_counts.items(), key=lambda x: -x[1]):
        print(f"  {cnt:6d}  {prov}")

    # ---- Plots ----
    print("\n" + "=" * 60)
    print("SAVING PLOTS")
    print("=" * 60)

    sorted_species = sorted(sp_counts.items(), key=lambda x: -x[1])
    top20_names    = [s for s, _ in sorted_species[:20]]
    top20_values   = [sp_counts[s] for s in top20_names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dataset Exploration", fontsize=14)

    # 1. Samples-per-species histogram
    axes[0, 0].hist(counts, bins=30, color="steelblue", edgecolor="white")
    axes[0, 0].set_title("Samples per Species Distribution")
    axes[0, 0].set_xlabel("Number of Samples")
    axes[0, 0].set_ylabel("Number of Species")

    # 2. Top 20 species
    axes[0, 1].barh(range(20), top20_values[::-1], color="steelblue")
    axes[0, 1].set_yticks(range(20))
    axes[0, 1].set_yticklabels(top20_names[::-1], fontsize=7)
    axes[0, 1].set_title("Top 20 Most Common Species")
    axes[0, 1].set_xlabel("Number of Samples")

    # 3. Month distribution
    months_ordered = list(range(1, 13))
    month_vals = [month_counts.get(m, 0) for m in months_ordered]
    axes[1, 0].bar([MONTH_NAMES[m] for m in months_ordered], month_vals,
                   color="coral", edgecolor="white")
    axes[1, 0].set_title("Observations by Month")
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Number of Samples")

    # 4. Geographic scatter (Netherlands)
    axes[1, 1].scatter(lons, lats, alpha=0.3, s=3, color="forestgreen")
    axes[1, 1].set_title("Geographic Distribution")
    axes[1, 1].set_xlabel("Longitude")
    axes[1, 1].set_ylabel("Latitude")
    axes[1, 1].set_xlim(3.3, 7.2)
    axes[1, 1].set_ylim(50.7, 53.6)

    plt.tight_layout()
    path = RESULTS_DIR / "data_exploration.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # Cleaning pipeline bar chart
    steps   = ["Raw rows", "Fields OK", "Has image", f">=MIN_GBIFIDS ({MIN_GBIFIDS})"]
    step_n  = [len(all_rows), len(has_fields), len(has_image), len(final)]
    fig2, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(steps, step_n, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%d", fontsize=9)
    ax.set_title("Cleaning Pipeline: Records Remaining per Step")
    ax.set_ylabel("Number of Records")
    plt.tight_layout()
    path2 = RESULTS_DIR / "cleaning_steps.png"
    plt.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"Saved: {path2}")

    log_file.close()
    sys.stdout = sys.__stdout__
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
