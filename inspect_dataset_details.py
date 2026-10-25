
from pathlib import Path
import pandas as pd

folder = Path.home() / "Downloads" / "raindrop_dataset"
files = sorted(folder.rglob("*.csv"))

if not files:
    raise FileNotFoundError(f"No CSV files found in {folder}")

# Inspect only the first CSV and first 10,000 rows
file = files[0]
df = pd.read_csv(file, comment="#", nrows=10_000)

print("FILE:", file.name)
print("\nCOLUMNS:")
print(df.columns.tolist())

print("\nDATA TYPES:")
print(df.dtypes)

print("\nFIRST FIVE ROWS:")
print(df.head().to_string(index=False))

print("\nSTATISTICAL SUMMARY:")
print(df.describe(include="all").to_string())

print("\nMISSING VALUES:")
print(df.isna().sum())

print("\nUNIQUE VALUES:")
print(df.nunique())
