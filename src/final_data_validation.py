
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# Dataset location
ZIP_PATH = Path("data/raindrop_volume_processed.zip")
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 200_000
REQUIRED_COLUMNS = {
    "DateTime",
    "Diameter_mm",
    "Velocity_m_s",
    "Volume_mm3",
    "Condition",
    "SourceFile",
}

def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Dataset ZIP not found: {ZIP_PATH.resolve()}"
        )

    print("Checking ZIP archive...")
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        csv_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
            and not name.startswith("__MACOSX/")
        ]

        if len(csv_files) != 1:
            raise ValueError(
                f"Expected one CSV in ZIP; found {len(csv_files)}: "
                f"{csv_files}"
            )

        csv_name = csv_files[0]
        print("CSV found:", csv_name)
        print("Archive integrity check...")
        bad_file = archive.testzip()

        if bad_file:
            raise ValueError(
                f"ZIP integrity check failed at: {bad_file}"
            )

    print("ZIP integrity: OK")
    print("\nReading CSV in chunks. This may take some time...")

    total_rows = 0
    missing_counts = None
    invalid_measurement_rows = 0
    formula_mismatch_rows = 0
    max_formula_difference = 0.0
    duplicate_rows_within_chunks = 0
    diameter_min = float("inf")
    diameter_max = float("-inf")
    velocity_min = float("inf")
    velocity_max = float("-inf")
    volume_min = float("inf")
    volume_max = float("-inf")
    chunk_number = 0

    # Read directly from the ZIP without extracting the huge CSV.
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        with archive.open(csv_name) as csv_stream:
            reader = pd.read_csv(
                csv_stream,
                chunksize=CHUNK_SIZE
            )

            for chunk in reader:
                chunk_number += 1
                total_rows += len(chunk)

                if missing_counts is None:
                    missing_columns = (
                        REQUIRED_COLUMNS - set(chunk.columns)
                    )
                    if missing_columns:
                        raise ValueError(
                            "Missing required columns: "
                            f"{sorted(missing_columns)}"
                        )
                    missing_counts = pd.Series(
                        0, index=chunk.columns, dtype="int64"
                    )

                missing_counts += chunk.isna().sum()

                diameter = pd.to_numeric(
                    chunk["Diameter_mm"], errors="coerce"
                )
                velocity = pd.to_numeric(
                    chunk["Velocity_m_s"], errors="coerce"
                )
                volume = pd.to_numeric(
                    chunk["Volume_mm3"], errors="coerce"
                )

                valid_numeric = (
                    np.isfinite(diameter)
                    & np.isfinite(velocity)
                    & np.isfinite(volume)
                )

                valid_positive = (
                    diameter.gt(0)
                    & velocity.gt(0)
                    & volume.gt(0)
                )

                invalid_measurement_rows += int(
                    (~(valid_numeric & valid_positive)).sum()
                )

                # Verify stored volume against the spherical-drop formula.
                expected_volume = np.pi / 6 * diameter ** 3
                differences = (volume - expected_volume).abs()
                comparable = (
                    valid_numeric
                    & np.isfinite(differences)
                )

                if comparable.any():
                    diff = differences[comparable]
                    formula_mismatch_rows += int(
                        (diff > 0.001).sum()
                    )
                    max_formula_difference = max(
                        max_formula_difference,
                        float(diff.max())
                    )

                # Exact duplicate rows within each individual chunk.
                duplicate_rows_within_chunks += int(
                    chunk.duplicated().sum()
                )

                if valid_numeric.any():
                    diameter_min = min(
                        diameter_min, float(diameter[valid_numeric].min())
                    )
                    diameter_max = max(
                        diameter_max, float(diameter[valid_numeric].max())
                    )
                    velocity_min = min(
                        velocity_min, float(velocity[valid_numeric].min())
                    )
                    velocity_max = max(
                        velocity_max, float(velocity[valid_numeric].max())
                    )
                    volume_min = min(
                        volume_min, float(volume[valid_numeric].min())
                    )
                    volume_max = max(
                        volume_max, float(volume[valid_numeric].max())
                    )

                print(
                    f"Processed chunk {chunk_number} | "
                    f"Rows checked: {total_rows:,}"
                )

    if total_rows == 0:
        raise ValueError("The CSV contains no data rows.")

    missing_counts.to_csv(
        OUTPUT_DIR / "full_dataset_missing_values.csv",
        header=["Missing_Count"]
    )

    summary = pd.DataFrame([
        ["Total rows scanned", total_rows],
        ["Invalid measurement rows", invalid_measurement_rows],
        ["Volume formula mismatches (> 0.001 mm3)",
         formula_mismatch_rows],
        ["Maximum formula difference (mm3)",
         max_formula_difference],
        ["Exact duplicate rows within chunks",
         duplicate_rows_within_chunks],
        ["Minimum diameter (mm)", diameter_min],
        ["Maximum diameter (mm)", diameter_max],
        ["Minimum velocity (m/s)", velocity_min],
        ["Maximum velocity (m/s)", velocity_max],
        ["Minimum volume (mm3)", volume_min],
        ["Maximum volume (mm3)", volume_max],
    ], columns=["Check", "Result"])

    summary.to_csv(
        OUTPUT_DIR / "full_dataset_validation_summary.csv",
        index=False
    )

    print("\n========== VALIDATION SUMMARY ==========")
    print(summary.to_string(index=False))

    print("\nMissing values by column:")
    print(missing_counts.to_string())

    print("\nSaved reports:")
    print("models/full_dataset_validation_summary.csv")
    print("models/full_dataset_missing_values.csv")

    print(
        "\nIMPORTANT: Duplicate detection here is within each chunk only. "
        "It does not detect duplicates spanning different chunks."
    )

if __name__ == "__main__":
    main()
