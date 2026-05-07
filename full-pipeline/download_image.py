import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import config


# ---- Configuration ----
IMAGE_DIR    = config.IMAGE_DIR
MAX_WORKERS  = 8
TIMEOUT      = 10


# ---- Download a single image ----
def download_image(row):
    gbif_id = row['gbifID']
    url     = row['identifier']
    save_path = os.path.join(IMAGE_DIR, f"{gbif_id}.jpg")

    # Skip files that are already present in the local cache.
    if os.path.exists(save_path):
        return gbif_id, True, "already exists"

    try:
        response = requests.get(url, timeout=TIMEOUT)
        image    = Image.open(BytesIO(response.content)).convert('RGB')
        image    = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        image.save(save_path, 'JPEG', quality=90)
        return gbif_id, True, "downloaded"
    except Exception as e:
        return gbif_id, False, str(e)


# ---- Main download script ----
if __name__ == '__main__':

    # Create image directory
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Load multimedia file
    print("Loading multimedia data...")
    med = pd.read_csv(config.MULTIMEDIA_FILE, sep='\t', low_memory=False,
                      usecols=['gbifID', 'identifier', 'format'])
    med = med[med['format'] == 'image/jpeg']
    med = med.groupby('gbifID').first().reset_index()

    # Load occurrence to filter adults only
    print("Loading occurrence data...")
    occ = pd.read_csv(config.OCCURRENCE_FILE, sep='\t', low_memory=False,
                      usecols=['gbifID', 'lifeStage', 'species'])
    occ = occ[occ['lifeStage'] == config.LIFE_STAGE]
    occ = occ.dropna(subset=['species'])

    # Merge to get only relevant images
    df = pd.merge(occ, med, on='gbifID', how='inner')
    print(f"Total images to download: {len(df)}")

    # Check already downloaded
    already = len([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])
    print(f"Already downloaded: {already}")
    print(f"Remaining: {len(df) - already}")

    # Download in parallel
    success = 0
    failed  = 0
    failed_ids = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_image, row): row for _, row in df.iterrows()}

        with tqdm(total=len(df), desc="Downloading images") as pbar:
            for future in as_completed(futures):
                gbif_id, ok, msg = future.result()
                if ok:
                    success += 1
                else:
                    failed += 1
                    failed_ids.append(gbif_id)
                pbar.update(1)

    print(f"\nDownload complete!")
    print(f"Success: {success}")
    print(f"Failed:  {failed}")

    # Save failed IDs for inspection
    if failed_ids:
        pd.DataFrame({'gbifID': failed_ids}).to_csv('failed_downloads.csv', index=False)
        print(f"Failed IDs saved to failed_downloads.csv")
