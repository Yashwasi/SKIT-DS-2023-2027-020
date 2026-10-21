
import zipfile
import pandas as pd

zip_path = r"C:\Users\shubh\Downloads\raindrop_volume_processed.zip"

total_rows = 0
missing_values = 0
source_files = set()
exact_duplicates = 0
seen_rows = set()

with zipfile.ZipFile(zip_path) as archive:
    with archive.open("raindrop_volume_dataset.csv") as file:
        for chunk in pd.read_csv(file, chunksize=200_000):
            total_rows += len(chunk)
            missing_values += int(chunk.isna().sum().sum())
            source_files.update(
                chunk["SourceFile"].dropna().unique()
            )

            # Count exact duplicate rows across the full dataset.
            # Keep only unique row hashes in memory.
            row_hashes = pd.util.hash_pandas_object(
                chunk, index=False
            )

            for row_hash in row_hashes:
                value = int(row_hash)
                if value in seen_rows:
                    exact_duplicates += 1
                else:
                    seen_rows.add(value)

print("Total rows:", total_rows)
print("Missing values:", missing_values)
print("Source files:", len(source_files))
print("Exact duplicate candidates:", exact_duplicates)
print("Rows after removing candidates:", total_rows - exact_duplicates)
