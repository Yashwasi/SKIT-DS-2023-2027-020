
from pathlib import Path
import pandas as pd

# Folder containing the dataset
DATA_DIR = Path("data/raindrop2_raw/raindrop2")

# Find all CSV files
csv_files = sorted(DATA_DIR.glob("*.csv"))

print(f"Total CSV files: {len(csv_files)}")

# Read all files and combine them into one table
dataframes = []

for file_path in csv_files:
    df = pd.read_csv(file_path, skiprows=1)

    condition = file_path.stem.split("[")[-1].rstrip("]")
    df = df.assign(Condition=condition)

    # Keep the condition name from the filename
    df["Condition"] = file_path.stem.split("[")[-1].rstrip("]")

    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

# Convert rainfall intensity to numeric values
data["RainRate(mm/h)"] = pd.to_numeric(
    data["RainRate(mm/h)"],
    errors="coerce",
)

# Basic analysis
rainfall = data["RainRate(mm/h)"]

print("\n--- Dataset Overview ---")
print(f"Total records: {len(data)}")
print(f"Missing rainfall values: {rainfall.isna().sum()}")

print("\n--- Rainfall Intensity Statistics ---")
print(rainfall.describe())

print("\n--- Rainfall Categories ---")
print(f"Zero rainfall records: {(rainfall == 0).sum()}")
print(f"Positive rainfall records: {(rainfall > 0).sum()}")
print(f"Negative rainfall records: {(rainfall < 0).sum()}")

print("\n--- Highest Rainfall Intensity ---")
print(data.loc[rainfall.idxmax(), ["D-M-Y", "Time", "Condition", "RainRate(mm/h)"]])
