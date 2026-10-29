
from pathlib import Path
import pandas as pd

# Location of the extracted dataset
DATA_FOLDER = Path.home() / "Downloads" / "raindrop_dataset"

print("RAINDROP DATASET VERIFICATION")
print("=" * 50)

files = sorted(DATA_FOLDER.rglob("*.csv"))

if not files:
    print("No CSV files found. Check the dataset folder path.")
    raise SystemExit

print("CSV files found:", len(files))

total_rows = 0
total_missing = 0
total_duplicates = 0

for file in files:
    print(f"\nChecking: {file.name}")

    # Read in chunks to handle large files
    file_rows = 0
    file_missing = 0
    file_duplicates = 0
    seen = set()

    for chunk in pd.read_csv(
        file,
        comment="#",
        chunksize=100_000
    ):
        file_rows += len(chunk)
        file_missing += int(chunk.isna().sum().sum())

        # Exact duplicate check within each file.
        # Hashes are used to avoid storing every full row.
        hashes = pd.util.hash_pandas_object(
            chunk, index=False
        )

        for value in hashes:
            value = int(value)
            if value in seen:
                file_duplicates += 1
            else:
                seen.add(value)

    total_rows += file_rows
    total_missing += file_missing
    total_duplicates += file_duplicates

    print("Rows:", f"{file_rows:,}")
    print("Missing values:", f"{file_missing:,}")
    print("Exact duplicate candidates:", f"{file_duplicates:,}")

print("\n" + "=" * 50)
print("TOTAL ROWS:", f"{total_rows:,}")
print("TOTAL MISSING VALUES:", f"{total_missing:,}")
print("WITHIN-FILE DUPLICATE CANDIDATES:",
      f"{total_duplicates:,}")
print("NOTE: Cross-file duplicates are not checked here.")
