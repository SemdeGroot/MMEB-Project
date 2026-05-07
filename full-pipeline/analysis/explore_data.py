import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import matplotlib.pyplot as plt
import pandas as pd

import config
from dataset import STATE_PROVINCE_NORMALIZATION, get_metadata_columns, preprocess_metadata


# ---- Directories and constants ----
RESULTS_ROOT = "results"
RESULTS_DIR = os.path.join(RESULTS_ROOT, "explore_data")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

OCCURRENCE_COLS = [
    'gbifID',
    'species',
    'decimalLatitude',
    'decimalLongitude',
    'month',
    'year',
    'day',
    'lifeStage',
    'eventTime',
    'stateProvince',
    'coordinateUncertaintyInMeters',
]

MULTIMEDIA_COLS = [
    'gbifID',
    'identifier',
    'format',
]

SELECTED_RAW_METADATA_FIELDS = [
    'decimalLatitude',
    'decimalLongitude',
    'month',
    'day',
    'year',
    'eventTime',
    'stateProvince',
    'coordinateUncertaintyInMeters',
]

MONTH_NAMES = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec',
}


# ---- Tee: print to both console and file ----
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


# ---- Small helpers ----
def print_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def format_pct(part, whole):
    if whole == 0:
        return "0.00%"
    return f"{(part / whole) * 100:.2f}%"


def apply_overrides(args):
    if args.occurrence_file:
        config.OCCURRENCE_FILE = args.occurrence_file
        config.DATA_DIR = os.path.dirname(args.occurrence_file)
    if args.multimedia_file:
        config.MULTIMEDIA_FILE = args.multimedia_file


def load_raw_data():
    print("Loading occurrence data...")
    occ = pd.read_csv(
        config.OCCURRENCE_FILE,
        sep='\t',
        low_memory=False,
        usecols=OCCURRENCE_COLS,
    )

    print("Loading multimedia data...")
    med = pd.read_csv(
        config.MULTIMEDIA_FILE,
        sep='\t',
        low_memory=False,
        usecols=MULTIMEDIA_COLS,
    )
    return occ, med


def load_all_occurrence_columns():
    # Read only the header so we can list every available occurrence field.
    header_df = pd.read_csv(
        config.OCCURRENCE_FILE,
        sep='\t',
        low_memory=False,
        nrows=0,
    )
    return list(header_df.columns)


def summarize_project_usage():
    print_section("PROJECT DATA USED")
    print("This script does not load every column in the raw files.")
    print("It loads only the columns used by the current project.")

    print("\nOccurrence columns used by this project:")
    for col in OCCURRENCE_COLS:
        print(f"  - {col}")

    print("\nMultimedia columns used by this project:")
    for col in MULTIMEDIA_COLS:
        print(f"  - {col}")

    print("\nCurrent cleaning rules:")
    print(f"  - Keep only lifeStage == {config.LIFE_STAGE}")
    print("  - Require non-missing species, decimalLatitude, decimalLongitude, month, identifier")
    print(f"  - Drop species with fewer than {config.MIN_SAMPLES_PER_CLASS} samples")
    if config.NUM_SAMPLES is None:
        print("  - Use all available samples after cleaning")
    else:
        print(f"  - Subsample to at most {config.NUM_SAMPLES} samples after cleaning")


def summarize_raw_tables(occ, med):
    print_section("RAW TABLE OVERVIEW")
    print(f"Occurrence rows:               {len(occ)}")
    print(f"Occurrence unique gbifID:      {occ['gbifID'].nunique()}")
    print(f"Occurrence unique species:     {occ['species'].dropna().nunique()}")
    print(f"Multimedia rows:               {len(med)}")
    print(f"Multimedia unique gbifID:      {med['gbifID'].nunique()}")
    print(f"Multimedia rows with URL:      {med['identifier'].notna().sum()}")

    duplicate_occ = int(occ.duplicated(subset=['gbifID']).sum())
    duplicate_med = int(med.duplicated(subset=['gbifID']).sum())
    print(f"Occurrence duplicated gbifID:  {duplicate_occ}")
    print(f"Multimedia duplicated gbifID:  {duplicate_med}")

    print("\nMissing values in occurrence columns used by the project:")
    occ_missing = occ[OCCURRENCE_COLS].isna().sum().sort_values(ascending=False)
    for col, count in occ_missing.items():
        print(f"  {col:30} {count:>10} ({format_pct(count, len(occ))})")

    print("\nMissing values in multimedia columns used by the project:")
    med_missing = med[MULTIMEDIA_COLS].isna().sum().sort_values(ascending=False)
    for col, count in med_missing.items():
        print(f"  {col:30} {count:>10} ({format_pct(count, len(med))})")


def prepare_multimedia(med):
    med = med.copy()
    med['format'] = med['format'].astype(str).str.lower()

    print_section("MULTIMEDIA OVERVIEW")
    format_counts = med['format'].fillna('Unknown').value_counts()
    print("Multimedia format distribution:")
    print(format_counts.head(15).to_string())

    jpeg_rows = med[med['format'] == 'image/jpeg']
    image_count_per_id = jpeg_rows.groupby('gbifID')['identifier'].count()
    if len(image_count_per_id) > 0:
        print("\nJPEG images per gbifID:")
        print(f"  Mean: {image_count_per_id.mean():.2f}")
        print(f"  Std:  {image_count_per_id.std():.2f}")
        print(f"  Min:  {image_count_per_id.min()}")
        print(f"  Max:  {image_count_per_id.max()}")
    else:
        print("\nNo JPEG rows found.")

    jpeg_first = jpeg_rows.groupby('gbifID').first().reset_index()
    print(f"\nJPEG multimedia rows:         {len(jpeg_rows)}")
    print(f"Unique gbifID with JPEG:      {jpeg_rows['gbifID'].nunique()}")
    print(f"Rows kept after first-image:  {len(jpeg_first)}")
    return jpeg_first


def build_cleaning_steps(occ, med_first):
    # Recreate the same filtering logic as the training pipeline, step by step.
    merged = pd.merge(occ, med_first, on='gbifID', how='inner')

    after_adult = merged[merged['lifeStage'] == config.LIFE_STAGE].copy()
    after_required = after_adult.dropna(
        subset=['species', 'decimalLatitude', 'decimalLongitude', 'month', 'identifier']
    ).copy()

    species_counts_before_threshold = after_required['species'].value_counts()
    valid_species = species_counts_before_threshold[
        species_counts_before_threshold >= config.MIN_SAMPLES_PER_CLASS
    ].index
    after_species_threshold = after_required[after_required['species'].isin(valid_species)].copy()

    final_df = after_species_threshold.copy()
    if config.NUM_SAMPLES is not None:
        sample_size = min(config.NUM_SAMPLES, len(final_df))
        final_df = final_df.sample(n=sample_size, random_state=config.RANDOM_SEED)

    steps = [
        ("Merged occurrence + first JPEG multimedia", len(merged)),
        (f"After keeping only {config.LIFE_STAGE}", len(after_adult)),
        ("After dropping rows missing required fields", len(after_required)),
        (
            f"After dropping species with < {config.MIN_SAMPLES_PER_CLASS} samples",
            len(after_species_threshold),
        ),
        ("Final dataset used by training", len(final_df)),
    ]

    return merged, after_required, final_df.reset_index(drop=True), species_counts_before_threshold, steps


def summarize_cleaning(merged, after_required, final_df, species_counts_before_threshold, steps):
    print_section("CLEANING PIPELINE")
    previous = None

    for label, count in steps:
        removed = 0 if previous is None else previous - count
        print(f"{label:52} {count:>10} rows", end="")
        if previous is not None:
            print(f" | removed {removed:>8} ({format_pct(removed, previous)})")
        else:
            print()
        previous = count

    species_kept = final_df['species'].nunique()
    species_before = after_required['species'].dropna().nunique()
    species_dropped = species_before - species_kept
    print(f"\nSpecies before minimum-sample filter: {species_before}")
    print(f"Species kept after filter:            {species_kept}")
    print(f"Species dropped by filter:            {species_dropped}")

    dropped_species = species_counts_before_threshold[
        species_counts_before_threshold < config.MIN_SAMPLES_PER_CLASS
    ]
    if len(dropped_species) > 0:
        print("\nExample species dropped by the minimum-sample rule:")
        print(dropped_species.head(20).to_string())


def summarize_final_dataset(final_df):
    print_section("FINAL DATASET OVERVIEW")
    print(f"Final rows used by project:       {len(final_df)}")
    print(f"Final species count:              {final_df['species'].nunique()}")
    print(f"Unique gbifID:                    {final_df['gbifID'].nunique()}")
    print(
        f"Unique exact locations:           "
        f"{final_df[['decimalLatitude', 'decimalLongitude']].drop_duplicates().shape[0]}"
    )

    year_series = pd.to_numeric(final_df['year'], errors='coerce')
    if year_series.notna().any():
        print(f"Year range:                       {int(year_series.min())} - {int(year_series.max())}")
    else:
        print("Year range:                       unavailable")

    print(
        f"Random-chance baseline:           "
        f"{100.0 / final_df['species'].nunique():.2f}%"
    )


def summarize_species(final_df):
    print_section("SPECIES DISTRIBUTION")
    species_counts = final_df['species'].value_counts()
    print(f"Samples per species (mean):       {species_counts.mean():.2f}")
    print(f"Samples per species (std):        {species_counts.std():.2f}")
    print(f"Samples per species (min):        {species_counts.min()}")
    print(f"Samples per species (max):        {species_counts.max()}")
    print("\nTop 20 most common species:")
    print(species_counts.head(20).to_string())
    print("\nTop 20 rarest kept species:")
    print(species_counts.tail(20).to_string())
    return species_counts


def summarize_time(final_df):
    print_section("TEMPORAL OVERVIEW")
    month_counts = final_df['month'].value_counts().sort_index()
    print("Month distribution:")
    for month, count in month_counts.items():
        label = MONTH_NAMES.get(int(month), str(month))
        print(f"  {label:>3}: {count}")

    event_time_missing = int(final_df['eventTime'].isna().sum())
    day_missing = int(final_df['day'].isna().sum())
    year_missing = int(final_df['year'].isna().sum())
    print(f"\nMissing eventTime rows:           {event_time_missing} ({format_pct(event_time_missing, len(final_df))})")
    print(f"Missing day rows:                 {day_missing} ({format_pct(day_missing, len(final_df))})")
    print(f"Missing year rows:                {year_missing} ({format_pct(year_missing, len(final_df))})")
    return month_counts


def summarize_geography(final_df):
    print_section("GEOGRAPHY AND PLACE NAMES")
    print(f"Latitude range:                   {final_df['decimalLatitude'].min():.4f} to {final_df['decimalLatitude'].max():.4f}")
    print(f"Longitude range:                  {final_df['decimalLongitude'].min():.4f} to {final_df['decimalLongitude'].max():.4f}")

    raw_places = (
        final_df['stateProvince']
        .fillna('Unknown')
        .astype(str)
        .str.strip()
        .replace('', 'Unknown')
    )
    place_counts = raw_places.value_counts()
    print(f"\nUnique raw stateProvince names:   {raw_places.nunique()}")
    print("\nAll raw stateProvince names and counts:")
    print(place_counts.to_string())

    print("\nConfigured stateProvince renames:")
    if not STATE_PROVINCE_NORMALIZATION:
        print("  None")
    else:
        for old, new in STATE_PROVINCE_NORMALIZATION.items():
            old_count = int((raw_places == old).sum())
            new_count = int((raw_places == new).sum())
            print(
                f"  {old} -> {new}: old={old_count}, "
                f"existing_target={new_count}, merged_total={old_count + new_count}"
            )

    return place_counts


def summarize_metadata(final_df):
    print_section("METADATA USED BY THE MODELS")
    all_occurrence_cols = load_all_occurrence_columns()

    print("All available raw occurrence columns (possible metadata candidates):")
    for idx, col in enumerate(all_occurrence_cols, start=1):
        print(f"  {idx:>3}. {col}")

    print("\nRaw metadata fields selected by the project:")
    for idx, field in enumerate(SELECTED_RAW_METADATA_FIELDS, start=1):
        print(f"  {idx:>2}. {field}")

    print("\nMissingness for the selected raw metadata fields:")
    for field in SELECTED_RAW_METADATA_FIELDS:
        missing = int(final_df[field].isna().sum())
        print(f"  - {field:28} missing={missing:>7} ({format_pct(missing, len(final_df))})")

    preprocessed = preprocess_metadata(final_df)
    metadata_cols = get_metadata_columns(preprocessed)

    print("\nFinal processed metadata features selected for the models:")
    for idx, col in enumerate(metadata_cols, start=1):
        print(f"  {idx:>2}. {col}")

    print("\nProcessed metadata feature ranges:")
    stats = preprocessed[metadata_cols].agg(['min', 'max', 'mean', 'std']).T
    for col, row in stats.iterrows():
        print(
            f"  {col:28} "
            f"min={row['min']:>8.4f} "
            f"max={row['max']:>8.4f} "
            f"mean={row['mean']:>8.4f} "
            f"std={row['std']:>8.4f}"
        )


def save_individual_plots(species_counts, month_counts, place_counts, final_df, steps):
    # Save the figures separately so they can be reused directly in the report.
    step_labels = ['Merged', 'Adult only', 'Required fields', 'Min samples', 'Final']
    step_counts = [count for _, count in steps]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(step_labels, step_counts, color='slateblue', edgecolor='white')
    ax.set_title('Rows Remaining After Cleaning Steps')
    ax.set_ylabel('Number of Rows')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'cleaning_steps.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(species_counts.values, bins=30, color='steelblue', edgecolor='white')
    ax.set_title('Samples per Species Distribution')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Number of Species')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'species_distribution.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    top20_species = species_counts.head(20)
    ax.barh(top20_species.index, top20_species.values, color='teal')
    ax.set_title('Top 20 Most Common Species')
    ax.invert_yaxis()
    ax.tick_params(axis='y', labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'top20_species.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [MONTH_NAMES.get(int(m), str(m)) for m in month_counts.index],
        month_counts.values,
        color='coral',
        edgecolor='white',
    )
    ax.set_title('Observations by Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Samples')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'month_distribution.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    top_places = place_counts.head(15)
    ax.barh(top_places.index, top_places.values, color='darkgreen')
    ax.set_title('Top 15 Raw stateProvince Names')
    ax.invert_yaxis()
    ax.tick_params(axis='y', labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'state_province_top15.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        final_df['decimalLongitude'],
        final_df['decimalLatitude'],
        alpha=0.25,
        s=5,
        color='firebrick',
    )
    ax.set_title('Geographic Distribution of Final Dataset')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'geographic_distribution.png'), dpi=150)
    plt.close(fig)


def create_plots(species_counts, month_counts, place_counts, final_df, steps):
    save_individual_plots(species_counts, month_counts, place_counts, final_df, steps)
    print(f"\nSaved individual figures to {FIGURES_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive dataset exploration for the moth project.")
    parser.add_argument('--occurrence-file', type=str, default=None,
                        help='Optional path to occurrence.txt')
    parser.add_argument('--multimedia-file', type=str, default=None,
                        help='Optional path to multimedia.txt')
    args = parser.parse_args()

    apply_overrides(args)

    log_path = os.path.join(RESULTS_DIR, 'log_explore_data.txt')
    log_file = open(log_path, 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)

    try:
        print_section("DATA SOURCES")
        print(f"Occurrence file: {config.OCCURRENCE_FILE}")
        print(f"Multimedia file: {config.MULTIMEDIA_FILE}")

        occ, med = load_raw_data()

        summarize_project_usage()
        summarize_raw_tables(occ, med)

        med_first = prepare_multimedia(med)
        merged, after_required, final_df, species_counts_before_threshold, steps = build_cleaning_steps(occ, med_first)
        summarize_cleaning(merged, after_required, final_df, species_counts_before_threshold, steps)
        summarize_final_dataset(final_df)

        species_counts = summarize_species(final_df)
        month_counts = summarize_time(final_df)
        place_counts = summarize_geography(final_df)
        summarize_metadata(final_df)

        create_plots(species_counts, month_counts, place_counts, final_df, steps)

        print_section("FILES SAVED")
        print(f"Main log: {log_path}")
        print(f"Log folder: {RESULTS_DIR}")
        print(f"Figures are in: {FIGURES_DIR}")
        print("\nDone.")
    finally:
        log_file.close()
        sys.stdout = sys.__stdout__
        print(f"Log saved to {log_path}")


if __name__ == '__main__':
    main()
