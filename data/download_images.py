#!/usr/bin/env python3
# Downloads all moth images from multimedia.txt and saves them to images/<species>/<id>.jpg
# Skips files that already exist so it is safe to re-run.
# Species filtering (minimum samples) happens in dataset.py, not here.

import csv
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA_DIR   = Path(__file__).parent
OCCURRENCE = DATA_DIR / "occurrence.txt"
MULTIMEDIA = DATA_DIR / "multimedia.txt"
IMAGES_DIR = DATA_DIR / "images"

TIMEOUT_S  = 15
MAX_WORKERS = 16
MAX_ERRORS  = 20   # stop after this many consecutive failures


def make_dirname(name):
    # strip author parentheses so the folder name stays short
    name = name.split("(")[0].strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:80]


def load_species_map():
    # maps each occurrence ID to a folder-safe species name
    species_map = {}
    with open(OCCURRENCE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sp  = row.get("scientificName", "").strip()
            gid = row.get("gbifID", "").strip()
            if sp and gid:
                species_map[gid] = make_dirname(sp)
    print(f"Loaded {len(species_map)} occurrence records")
    return species_map


def build_tasks(species_map):
    with open(MULTIMEDIA, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"Total multimedia rows: {len(rows)}")

    tasks = []
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
        species_dir = IMAGES_DIR / species_map[gid]
        photo_id = url.rstrip("/").split("/")[-2] if "/photos/" in url else url.split("/")[-1].split(".")[0]
        dest = species_dir / f"{gid}_{photo_id}.jpg"
        tasks.append((gid, url, species_dir, dest))
    return tasks


def download_one(args):
    gid, url, species_dir, dest = args
    if dest.exists():
        return "skipped"
    try:
        species_dir.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "MMEB-research/1.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            dest.write_bytes(resp.read())
        return "downloaded"
    except Exception as e:
        print(f"  failed {dest.name}: {e}", file=sys.stderr)
        return "error"


def download_images(tasks):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    total      = len(tasks)
    downloaded = skipped = errors = done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(download_one, t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            done  += 1
            if result == "downloaded":
                downloaded += 1
            elif result == "skipped":
                skipped += 1
            else:
                errors += 1
            if done % 100 == 0 or done == total:
                print(f"  {done}/{total}  downloaded={downloaded}  "
                      f"skipped={skipped}  errors={errors}", flush=True)
            if errors >= MAX_ERRORS:
                print(f"Stopping after {MAX_ERRORS} errors.", file=sys.stderr)
                ex.shutdown(wait=False, cancel_futures=True)
                break

    print(f"\nDone. total={total}, downloaded={downloaded}, "
          f"skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    species_map = load_species_map()
    tasks       = build_tasks(species_map)
    download_images(tasks)
