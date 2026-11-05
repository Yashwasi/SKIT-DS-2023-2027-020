
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Dataset folder
data_dir = Path("data/raindrop2_raw/raindrop2")

# Find all CSV files
csv_files = sorted(data_dir.glob("*.csv"))

# Read and combine rainfall measurements
frames = []

for file_path in csv_files:
    df = pd.read_csv(file_path, skiprows=1)
    frames.append(df[["RainRate(mm/h)"]])

data = pd.concat(frames, ignore_index=True)

# Convert rainfall intensity into numeric values
rainfall = pd.to_numeric(
    data["RainRate(mm/h)"],
    errors="coerce",
).dropna()

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(rainfall, bins=30)

plt.title("Distribution of Rainfall Intensity")
plt.xlabel("Rainfall Intensity (mm/h)")
plt.ylabel("Number of Observations")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
