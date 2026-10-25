"""
handle_missing_values.py
------------------------
User Story: To Handle missing values (06/09/2026 – 03/10/2026)
Purpose   : Detect, report and fix missing / inconsistent values in the
            deduplicated raindrop dataset.

Strategy
--------
1. Load the deduplicated CSV (sampled for speed; full file supported).
2. Report any NULL / NaN values per column.
3. Remove rows where any of the three physical measurements
   (Diameter_mm, Velocity_m_s, Volume_mm3) is NULL, zero or negative.
4. Impute remaining NULLs in numeric columns with the column median.
5. Fill missing Condition / SourceFile strings with 'Unknown'.
6. Re-validate the volume formula:  V = (π/6) × d³.
   Flag rows whose recorded volume deviates from the formula by more
   than VOLUME_TOLERANCE and set Volume_mm3 to the computed value.
7. Save the cleaned dataset and a short summary report.

Run from the repository root:
    python src/handle_missing_values.py
"""

from pathlib import Path
import math

import pandas as pd


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "raindrop_volume_processed"
    / "raindrop_volume_deduplicated.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "raindrop_volume_processed"
    / "raindrop_volume_clean.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = REPORT_DIR / "missing_values_report.md"

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

# Maximum absolute difference between stored and computed volume (mm³)
VOLUME_TOLERANCE = 0.001

# Number of rows to process; set to None to process the full dataset.
SAMPLE_SIZE = None  # e.g. 500_000 for a quick run


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def spherical_volume(diameter_mm: pd.Series) -> pd.Series:
    """Return the theoretical spherical volume for each diameter value."""
    return (math.pi / 6.0) * diameter_mm ** 3


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("RAIN DROP VOLUME PREDICTION USING AI")
    print("Missing-Value & Inconsistency Handler")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Deduplicated dataset not found:\n{DATA_PATH}\n"
            "Run deduplicate_dataset.py first."
        )

    print(f"\nLoading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, nrows=SAMPLE_SIZE)
    total_initial = len(df)
    print(f"Rows loaded: {total_initial:,}")
    print(f"Columns    : {list(df.columns)}")

    # ------------------------------------------------------------------
    # 2. Baseline missing-value report
    # ------------------------------------------------------------------
    null_counts_before = df.isnull().sum()
    print("\nNull values per column (before cleaning):")
    print(null_counts_before.to_string())

    # ------------------------------------------------------------------
    # 3. Drop rows with NULL / non-positive physical measurements
    # ------------------------------------------------------------------
    physical_cols = ["Diameter_mm", "Velocity_m_s", "Volume_mm3"]

    # Cast to numeric, coerce errors → NaN
    for col in physical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mask_invalid = (
        df[physical_cols].isnull().any(axis=1)
        | (df[physical_cols] <= 0).any(axis=1)
    )
    n_invalid = int(mask_invalid.sum())
    df = df[~mask_invalid].copy()
    print(
        f"\nDropped {n_invalid:,} rows with NULL / non-positive "
        "measurements."
    )

    # ------------------------------------------------------------------
    # 4. Median-impute remaining numeric NULLs
    # ------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    imputed_cols = []

    for col in numeric_cols:
        n_null = int(df[col].isnull().sum())
        if n_null > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            imputed_cols.append((col, n_null, median_val))
            print(
                f"Imputed {n_null:,} nulls in '{col}' "
                f"with median={median_val:.6f}"
            )

    # ------------------------------------------------------------------
    # 5. Fill categorical NULLs
    # ------------------------------------------------------------------
    for col in ["Condition", "SourceFile"]:
        if col in df.columns:
            n_null = int(df[col].isnull().sum())
            if n_null > 0:
                df[col] = df[col].fillna("Unknown")
                print(f"Filled {n_null:,} nulls in '{col}' with 'Unknown'.")

    # ------------------------------------------------------------------
    # 6. Volume formula validation & correction
    # ------------------------------------------------------------------
    computed_volume = spherical_volume(df["Diameter_mm"])
    volume_error = (df["Volume_mm3"] - computed_volume).abs()
    n_formula_mismatch = int((volume_error > VOLUME_TOLERANCE).sum())

    if n_formula_mismatch > 0:
        df.loc[
            volume_error > VOLUME_TOLERANCE, "Volume_mm3"
        ] = computed_volume[volume_error > VOLUME_TOLERANCE]
        print(
            f"\nCorrected {n_formula_mismatch:,} rows where recorded "
            f"volume deviated from V=(π/6)×d³ by more than "
            f"{VOLUME_TOLERANCE} mm³."
        )
    else:
        print(
            "\nAll volume values are consistent with the spherical "
            f"formula within {VOLUME_TOLERANCE} mm³ tolerance."
        )

    # ------------------------------------------------------------------
    # 7. Final null-value check
    # ------------------------------------------------------------------
    null_counts_after = df.isnull().sum()
    total_final = len(df)

    print("\nNull values per column (after cleaning):")
    print(null_counts_after.to_string())
    print(f"\nRows retained: {total_final:,} / {total_initial:,}")

    # ------------------------------------------------------------------
    # 8. Save cleaned dataset
    # ------------------------------------------------------------------
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()  # overwrite allowed for this cleaning pass

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCleaned dataset saved: {OUTPUT_CSV}")

    # ------------------------------------------------------------------
    # 9. Write markdown report
    # ------------------------------------------------------------------
    lines = []

    def add(text: str = "") -> None:
        lines.append(str(text))

    add("# Rain Drop Volume Prediction Using AI")
    add("## Data Cleaning — Missing Values & Inconsistency Report")
    add()
    add(f"**Input file:** `{DATA_PATH.name}`")
    add(f"**Output file:** `{OUTPUT_CSV.name}`")
    add()

    add("## 1. Null Values Before Cleaning")
    add()
    add("| Column | Null count |")
    add("|---|---:|")
    for col, cnt in null_counts_before.items():
        add(f"| {col} | {cnt:,} |")
    add()

    add("## 2. Invalid Row Removal")
    add()
    add(
        f"- Rows removed (NULL / non-positive measurements): "
        f"**{n_invalid:,}**"
    )
    add()

    add("## 3. Median Imputation")
    add()
    if imputed_cols:
        add("| Column | Nulls imputed | Median value used |")
        add("|---|---:|---:|")
        for col, n, med in imputed_cols:
            add(f"| {col} | {n:,} | {med:.6f} |")
    else:
        add("No median imputation was required.")
    add()

    add("## 4. Volume Formula Corrections")
    add()
    add(
        "- Formula: `Volume_mm3 = (π / 6) × Diameter_mm³`"
    )
    add(
        f"- Tolerance: `{VOLUME_TOLERANCE}` mm³"
    )
    add(
        f"- Rows corrected: **{n_formula_mismatch:,}**"
    )
    add()

    add("## 5. Null Values After Cleaning")
    add()
    add("| Column | Null count |")
    add("|---|---:|")
    for col, cnt in null_counts_after.items():
        add(f"| {col} | {cnt:,} |")
    add()

    add("## 6. Row Count Summary")
    add()
    add("| | Rows |")
    add("|---|---:|")
    add(f"| Initial rows | {total_initial:,} |")
    add(f"| Rows removed (invalid) | {n_invalid:,} |")
    add(f"| Rows retained | {total_final:,} |")
    add()

    add("---")
    add("*Generated automatically by `src/handle_missing_values.py`.*")

    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved: {REPORT_FILE}")

    print("\n" + "=" * 60)
    print("MISSING-VALUE HANDLING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
