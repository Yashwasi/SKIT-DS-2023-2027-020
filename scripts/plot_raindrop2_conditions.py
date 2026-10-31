
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Dataset folder
data_dir = Path("data/raindrop2_raw/raindrop2")

# Find all CSV files
csv_files = sorted(data_dir.glob("*.csv"))

# Read each file and calculate its rainfall statistics
results = []

for file_path in csv_files:
    df = pd.read_csv(file_path, skiprows=1)

    condition = file_path.stem.split("[")[-1].rstrip("]")

    rainfall = pd.to_numeric(
        df["RainRate(mm/h)"],
        errors="coerce",
    )

    results.append({
        "Condition": condition,
        "Average": rainfall.mean(),
    })

# Create a summary table
summary = pd.DataFrame(results)

# Select the 15 conditions with the highest averages
top_15 = summary.nlargest(15, "Average")
top_15 = top_15.sort_values("Average")

# Create the bar chart
plt.figure(figsize=(12, 7))

plt.barh(
    top_15["Condition"],
    top_15["Average"],
)

plt.title("Top 15 Conditions by Average Rainfall Intensity")
plt.xlabel("Average Rainfall Intensity (mm/h)")
plt.ylabel("Experimental Condition")

plt.tight_layout()
plt.show()
