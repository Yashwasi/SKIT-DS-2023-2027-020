
from pathlib import Path
import pandas as pd

folder = Path.home() / "Downloads" / "raindrop_dataset"
files = sorted(folder.rglob("*.csv"))

total_rows = 0
diameter_min = float("inf")
diameter_max = float("-inf")
velocity_min = float("inf")
velocity_max = float("-inf")

print("RAINDROP DATASET SUMMARY")
print("=" * 60)

for file in files:
    file_rows = 0

    try:
        for chunk in pd.read_csv(
            file,
            comment="#",
            chunksize=100_000
        ):
            file_rows += len(chunk)

            diameter_min = min(
                diameter_min, chunk["Diameter"].min()
            )
            diameter_max = max(
                diameter_max, chunk["Diameter"].max()
            )
            velocity_min = min(
                velocity_min, chunk["Velocity"].min()
            )
            velocity_max = max(
                velocity_max, chunk["Velocity"].max()
            )

        total_rows += file_rows
        print(f"{file.name}: {file_rows:,} rows")

    except Exception as e:
        print(f"Error in {file.name}: {e}")

print("=" * 60)
print(f"Total files found: {len(files)}")
print(f"Total rows processed: {total_rows:,}")
print(f"Diameter range: {diameter_min} to {diameter_max}")
print(f"Velocity range: {velocity_min} to {velocity_max}")
