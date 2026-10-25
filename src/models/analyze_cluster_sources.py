
from pathlib import Path

import pandas as pd


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "raindrop_volume_processed"
    / "raindrop_volume_deduplicated.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "cluster_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSIGNMENTS_FILE = OUTPUT_DIR / "cluster_assignments.csv"

SOURCE_REPORT = OUTPUT_DIR / "cluster_source_summary.csv"
DATE_REPORT = OUTPUT_DIR / "cluster_date_summary.csv"
TIME_REPORT = OUTPUT_DIR / "cluster_hour_summary.csv"

TARGET_CLUSTER = 2
CHUNK_SIZE = 500_000


# ==========================================
# 2. VALIDATE INPUT FILES
# ==========================================

def validate_files():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    if not ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Cluster assignments not found: {ASSIGNMENTS_FILE}"
        )


# ==========================================
# 3. LOAD CLUSTER 2 OBSERVATIONS
# ==========================================

def load_target_cluster():

    print("Loading Cluster 2 assignments...")

    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        usecols=[
            "Diameter_mm",
            "Velocity_m_s",
            "Cluster",
        ],
    )

    assignments["Cluster"] = pd.to_numeric(
        assignments["Cluster"],
        errors="coerce",
    )

    target = assignments[
        assignments["Cluster"] == TARGET_CLUSTER
    ].copy()

    print(f"Cluster 2 observations: {len(target):,}")

    if target.empty:
        raise ValueError(
            "No observations found for the selected cluster."
        )

    # Keep a count of each feature pair.
    target["Diameter_mm"] = pd.to_numeric(
        target["Diameter_mm"],
        errors="coerce",
    )

    target["Velocity_m_s"] = pd.to_numeric(
        target["Velocity_m_s"],
        errors="coerce",
    )

    target = target.dropna(
        subset=["Diameter_mm", "Velocity_m_s"]
    )

    pair_counts = (
        target.groupby(
            ["Diameter_mm", "Velocity_m_s"]
        )
        .size()
        .rename("Assignment_Count")
        .reset_index()
    )

    return target, pair_counts


# ==========================================
# 4. MATCH SOURCE RECORDS IN CHUNKS
# ==========================================

def find_source_records(pair_counts):

    print("\nSearching source dataset in chunks...")

    required_columns = [
        "DateTime",
        "Diameter_mm",
        "Velocity_m_s",
        "Condition",
        "SourceFile",
    ]

    available_columns = pd.read_csv(
        DATA_FILE,
        nrows=0,
    ).columns.tolist()

    missing = [
        col for col in required_columns
        if col not in available_columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    # Feature pairs from the saved cluster assignments.
    target_pairs = set(
        zip(
            pair_counts["Diameter_mm"].round(6),
            pair_counts["Velocity_m_s"].round(6),
        )
    )

    matches = []
    rows_scanned = 0

    for chunk in pd.read_csv(
        DATA_FILE,
        usecols=required_columns,
        chunksize=CHUNK_SIZE,
    ):

        rows_scanned += len(chunk)

        chunk["Diameter_mm"] = pd.to_numeric(
            chunk["Diameter_mm"],
            errors="coerce",
        )

        chunk["Velocity_m_s"] = pd.to_numeric(
            chunk["Velocity_m_s"],
            errors="coerce",
        )

        chunk = chunk.dropna(
            subset=["Diameter_mm", "Velocity_m_s"]
        )

        pairs = zip(
            chunk["Diameter_mm"].round(6),
            chunk["Velocity_m_s"].round(6),
        )

        mask = pd.Series(
            [pair in target_pairs for pair in pairs],
            index=chunk.index,
        )

        matched = chunk.loc[mask].copy()

        if not matched.empty:
            matches.append(matched)

        if rows_scanned % 2_000_000 == 0:
            print(f"Rows scanned: {rows_scanned:,}")

    if matches:
        result = pd.concat(matches, ignore_index=True)
    else:
        result = pd.DataFrame(columns=required_columns)

    print(f"\nDataset rows scanned: {rows_scanned:,}")
    print(f"Matching records found: {len(result):,}")

    return result


# ==========================================
# 5. CREATE SOURCE AND TIME REPORTS
# ==========================================

def generate_reports(records):

    if records.empty:
        print("No matching records found.")
        return

    records["DateTime"] = pd.to_datetime(
        records["DateTime"],
        errors="coerce",
    )

    records["Date"] = records["DateTime"].dt.strftime(
        "%Y-%m-%d"
    )

    records["Hour"] = records["DateTime"].dt.hour

    # Source-file distribution.
    source_summary = (
        records.groupby("SourceFile")
        .agg(
            Observation_Count=("Diameter_mm", "size"),
            Mean_Diameter=("Diameter_mm", "mean"),
            Max_Diameter=("Diameter_mm", "max"),
            Mean_Velocity=("Velocity_m_s", "mean"),
            Max_Velocity=("Velocity_m_s", "max"),
        )
        .reset_index()
        .sort_values(
            "Observation_Count",
            ascending=False,
        )
    )

    source_summary.to_csv(
        SOURCE_REPORT,
        index=False,
    )

    # Date distribution.
    date_summary = (
        records.groupby("Date", dropna=False)
        .agg(
            Observation_Count=("Diameter_mm", "size"),
            Mean_Diameter=("Diameter_mm", "mean"),
            Max_Diameter=("Diameter_mm", "max"),
            Mean_Velocity=("Velocity_m_s", "mean"),
            Max_Velocity=("Velocity_m_s", "max"),
        )
        .reset_index()
        .sort_values(
            "Observation_Count",
            ascending=False,
        )
    )

    date_summary.to_csv(
        DATE_REPORT,
        index=False,
    )

    # Hour-of-day distribution.
    hour_summary = (
        records.groupby("Hour", dropna=False)
        .agg(
            Observation_Count=("Diameter_mm", "size"),
            Mean_Diameter=("Diameter_mm", "mean"),
            Max_Diameter=("Diameter_mm", "max"),
            Mean_Velocity=("Velocity_m_s", "mean"),
            Max_Velocity=("Velocity_m_s", "max"),
        )
        .reset_index()
        .sort_values("Hour")
    )

    hour_summary.to_csv(
        TIME_REPORT,
        index=False,
    )

    print("\nTOP SOURCE FILES")
    print("-" * 70)
    print(source_summary.head(15).round(4).to_string(index=False))

    print("\nDATE DISTRIBUTION")
    print("-" * 70)
    print(date_summary.round(4).to_string(index=False))

    print("\nHOUR DISTRIBUTION")
    print("-" * 70)
    print(hour_summary.round(4).to_string(index=False))

    print("\nReports saved:")
    print(SOURCE_REPORT)
    print(DATE_REPORT)
    print(TIME_REPORT)


# ==========================================
# 6. MAIN
# ==========================================

def main():

    print("=" * 65)
    print("CLUSTER 2 SOURCE AND TIMESTAMP ANALYSIS")
    print("=" * 65)

    validate_files()

    target, pair_counts = load_target_cluster()

    records = find_source_records(pair_counts)

    generate_reports(records)

    print("\nSource analysis completed.")


if __name__ == "__main__":
    main()
