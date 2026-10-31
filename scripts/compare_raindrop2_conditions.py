
from pathlib import Path

import pandas as pd

# Dataset folder
data_dir = Path("data/raindrop2_raw/raindrop2")

# Find all CSV files
csv_files = sorted(data_dir.glob("*.csv"))

# Read rainfall data from every file
frames = []

for file_path in csv_files:
    df = pd.read_csv(file_path, skiprows=1)

    # Extract experimental condition from filename
    condition = file_path.stem.split("[")[-1].rstrip("]")

    # Keep only the required columns
    df = df[["RainRate(mm/h)"]].copy()
    df["Condition"] = condition

    frames.append(df)

# Combine all files
data = pd.concat(frames, ignore_index=True)

# Ensure rainfall intensity is numeric
data["RainRate(mm/h)"] = pd.to_numeric(
    data["RainRate(mm/h)"],
    errors="coerce",
)

# Calculate statistics for each condition
summary = (
    data.groupby("Condition")["RainRate(mm/h)"]
    .agg(
        Records="count",
        Average="mean",
        Median="median",
        Maximum="max",
        Zero_Rainfall=lambda values: (values == 0).sum(),
    )
    .reset_index()
)

# Show conditions with highest average intensity
summary = summary.sort_values("Average", ascending=False)

print("\n--- Rainfall Statistics by Condition ---")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print(f"\nTotal conditions: {summary['Condition'].nunique()}")
