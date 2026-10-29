
from pathlib import Path

import pandas as pd
import numpy as np


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "raindrop_volume_processed"
    / "raindrop_volume_deduplicated.csv"
)

EXTREME_FILE = (
    PROJECT_ROOT
    / "reports"
    / "cluster_analysis"
    / "extreme_observations.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "cluster_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "extreme_observations_with_source.csv"

FEATURES = ["Diameter_mm", "Velocity_m_s"]

METADATA_COLUMNS = ["DateTime", "Condition", "SourceFile"]


# ==========================================
# 2. VERIFY INPUT FILES
# ==========================================

def check_files():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Deduplicated dataset not found: {DATA_FILE}"
        )

    if not EXTREME_FILE.exists():
        raise FileNotFoundError(
            f"Extreme observations file not found: {EXTREME_FILE}"
        )


# ==========================================
# 3. LOAD EXTREME OBSERVATIONS
# ==========================================

def load_extreme_observations():

    print("Loading extreme observations...")

    extreme_df = pd.read_csv(EXTREME_FILE)

    for column in FEATURES:
        extreme_df[column] = pd.to_numeric(
            extreme_df[column],
            errors="coerce",
        )

    extreme_df = extreme_df.dropna(subset=FEATURES)

    # The report currently contains at most 100 selected rows.
    # We will inspect these exact diameter/velocity combinations.
    targets = extreme_df[FEATURES].drop_duplicates()

    print(f"Extreme rows loaded: {len(extreme_df):,}")
    print(f"Unique feature combinations: {len(targets):,}")

    return extreme_df, targets


# ==========================================
# 4. SEARCH ORIGINAL DATASET IN CHUNKS
# ==========================================

def find_source_records(targets):

    print("\nSearching the deduplicated dataset in chunks...")

    required_columns = FEATURES + METADATA_COLUMNS

    available_columns = pd.read_csv(
        DATA_FILE,
        nrows=0,
    ).columns.tolist()

    missing_columns = [
        column
        for column in required_columns
        if column not in available_columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns missing from dataset: "
            + ", ".join(missing_columns)
        )

    target_pairs = set(
        zip(
            targets["Diameter_mm"].round(6),
            targets["Velocity_m_s"].round(6),
        )
    )

    matched_chunks = []
    rows_scanned = 0

    for chunk in pd.read_csv(
        DATA_FILE,
        usecols=required_columns,
        chunksize=500_000,
    ):

        rows_scanned += len(chunk)

        for column in FEATURES:
            chunk[column] = pd.to_numeric(
                chunk[column],
                errors="coerce",
            )

        chunk = chunk.dropna(subset=FEATURES)

        pairs = list(
            zip(
                chunk["Diameter_mm"].round(6),
                chunk["Velocity_m_s"].round(6),
            )
        )

        mask = pd.Series(
            [pair in target_pairs for pair in pairs],
            index=chunk.index,
        )

        matched = chunk.loc[mask].copy()

        if not matched.empty:
            matched_chunks.append(matched)

        if rows_scanned % 2_000_000 == 0:
            print(f"Rows scanned: {rows_scanned:,}")

    if matched_chunks:
        source_records = pd.concat(
            matched_chunks,
            ignore_index=True,
        )
    else:
        source_records = pd.DataFrame(
            columns=required_columns
        )

    print(f"\nTotal dataset rows scanned: {rows_scanned:,}")
    print(f"Matching source rows found: {len(source_records):,}")

    return source_records


# ==========================================
# 5. SAVE INVESTIGATION REPORT
# ==========================================

def save_report(extreme_df, source_records):

    # Keep the cluster and outlier flags from the original report.
    flags = [
        column
        for column in extreme_df.columns
        if column not in FEATURES
    ]

    if source_records.empty:
        print(
            "\nNo matching source rows found. "
            "Check numeric precision and dataset contents."
        )
        return

    # Exact feature pairs may occur more than once in the source.
    # Preserve all matching source records rather than selecting one.
    report = source_records.merge(
        extreme_df[FEATURES + flags].drop_duplicates(),
        on=FEATURES,
        how="left",
    )

    report.to_csv(OUTPUT_FILE, index=False)

    print("\nSOURCE INVESTIGATION SUMMARY")
    print("=" * 65)

    print(
        report[
            FEATURES + METADATA_COLUMNS
        ].head(15).to_string(index=False)
    )

    print("\nReport saved to:")
    print(OUTPUT_FILE)

    print(
        "\nNote: Matching uses rounded feature values. "
        "Repeated feature combinations may match multiple source rows."
    )


# ==========================================
# 6. MAIN
# ==========================================

def main():

    print("=" * 65)
    print("EXTREME OBSERVATION SOURCE INVESTIGATION")
    print("=" * 65)

    check_files()

    extreme_df, targets = load_extreme_observations()

    source_records = find_source_records(targets)

    save_report(extreme_df, source_records)

    print("\nInvestigation completed.")


if __name__ == "__main__":
    main()

