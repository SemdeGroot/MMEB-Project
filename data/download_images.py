#!/usr/bin/env python3
# Downloads moth images from multimedia.txt and saves them to images/<species>/<id>.jpg
# Skips files that already exist so it is safe to re-run.

import csv
import re
import sys
import time
import urllib.request
from pathlib import Path

DATA_DIR   = Path(__file__).parent
OCCURRENCE = DATA_DIR / "occurrence.txt"
MULTIMEDIA = DATA_DIR / "multimedia.txt"
IMAGES_DIR = DATA_DIR / "images"

MIN_SAMPLES = 50   # skip species with fewer records than this
DELAY_S     = 0.05
TIMEOUT_S   = 15
MAX_ERRORS  = 20   # stop after this many consecutive failures


def make_dirname(name):
    # strip author parentheses so the folder name stays short
    name = name.split("(")[0].strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:80]


def load_species_map():
    counts = {}
    gbif_to_species = {}

    with open(OCCURRENCE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sp  = row.get("scientificName", "").strip()
            gid = row.get("gbifID", "").strip()
            if sp and gid:
                counts[sp] = counts.get(sp, 0) + 1
                gbif_to_species[gid] = sp

    allowed = {sp for sp, n in counts.items() if n >= MIN_SAMPLES}
    print(f"Species with >={MIN_SAMPLES} records: {len(allowed)}")

    return {
        gid: make_dirname(sp)
        for gid, sp in gbif_to_species.items()
        if sp in allowed
    }


def download_images(species_map):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    total = skipped = downloaded = errors = 0
    consecutive_errors = 0

    with open(MULTIMEDIA, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    print(f"Total multimedia rows: {len(rows)}")

    for row in rows:
        gid = row.get("gbifID", "").strip()
        url = row.get("identifier", "").strip()
        fmt = row.get("format", "").strip()

        if not url or not gid:
            continue
        if fmt and fmt != "image/jpeg":
            continue
        if gid not in species_map:
            continue

        total += 1
        species_dir = IMAGES_DIR / species_map[gid]
        species_dir.mkdir(exist_ok=True)

        # iNaturalist URLs look like .../photos/<photo_id>/original.jpg
        photo_id = url.rstrip("/").split("/")[-2] if "/photos/" in url else url.split("/")[-1].split(".")[0]
        dest = species_dir / f"{gid}_{photo_id}.jpg"

        if dest.exists():
            skipped += 1
            continue

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MMEB-research/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                dest.write_bytes(resp.read())
            downloaded += 1
            consecutive_errors = 0
            if downloaded % 100 == 0:
                print(f"  downloaded {downloaded}, skipped {skipped}, errors {errors}", flush=True)
            time.sleep(DELAY_S)
        except Exception as e:
            errors += 1
            consecutive_errors += 1
            print(f"  failed {dest.name}: {e}", file=sys.stderr)
            if consecutive_errors >= MAX_ERRORS:
                print(f"Stopping after {MAX_ERRORS} consecutive failures.", file=sys.stderr)
                break

    print(f"\nDone. total={total}, downloaded={downloaded}, skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    species_map = load_species_map()
    download_images(species_map)
