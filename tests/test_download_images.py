import sys
import csv
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.download_images import make_dirname, load_species_map

DATA_DIR   = Path(__file__).parent.parent / "data"
MULTIMEDIA = DATA_DIR / "multimedia.txt"
IMAGES_DIR = DATA_DIR / "images"

N_IMAGES = 50  # download until we have at least this many images


def ensure_sample_images():
    """Download images until we have at least N_IMAGES locally. Idempotent."""
    existing = list(IMAGES_DIR.glob("*/*.jpg"))
    if len(existing) >= N_IMAGES:
        print(f"Found {len(existing)} existing images, skipping download")
        return

    print(f"Have {len(existing)} images, downloading until we reach {N_IMAGES}...")
    species_map = load_species_map()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    with open(MULTIMEDIA, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if len(existing) + downloaded >= N_IMAGES:
                break

            gid = row.get("gbifID", "").strip()
            url = row.get("identifier", "").strip()
            fmt = row.get("format", "").strip()

            if not url or not gid or fmt != "image/jpeg" or gid not in species_map:
                continue

            species_dir = IMAGES_DIR / species_map[gid]
            species_dir.mkdir(exist_ok=True)

            photo_id = url.rstrip("/").split("/")[-2] if "/photos/" in url else url.split("/")[-1].split(".")[0]
            dest = species_dir / f"{gid}_{photo_id}.jpg"

            if dest.exists():
                continue

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "MMEB-research/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    dest.write_bytes(resp.read())
                downloaded += 1
                print(f"  downloaded {downloaded}: {dest.name}")
            except Exception as e:
                print(f"  skipped: {e}")

    total = len(list(IMAGES_DIR.glob("*/*.jpg")))
    print(f"Done, {total} images available")
    return total


print("--- download images ---")
total = ensure_sample_images()
total = total or len(list(IMAGES_DIR.glob("*/*.jpg")))
assert total >= N_IMAGES, f"Expected at least {N_IMAGES} images, got {total}"

# verify images are readable
sample = next(IMAGES_DIR.glob("*/*.jpg"))
assert sample.stat().st_size > 0, f"Image file is empty: {sample}"
print(f"Sample image: {sample} ({sample.stat().st_size} bytes)")

print("\nDownload test passed")
