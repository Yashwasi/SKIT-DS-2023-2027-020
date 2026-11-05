
from pathlib import Path
import pandas as pd
import numpy as np

folder = Path.home() / "Downloads" / "raindrop_dataset"
files = sorted(folder.rglob("*.csv"))

if not files:
    raise FileNotFoundError(f"No CSV files found in {folder}")

# Inspect a sample from the first source file
df = pd.read_csv(files[0], comment="#", nrows=50_000)

print("SOURCE:", files[0].name)
print("\nDIAMETER QUANTILES:")
print(df["Diameter"].quantile(
    [0, 0.01, 0.25, 0.50, 0.75, 0.99, 1]
))

print("\nVELOCITY QUANTILES:")
print(df["Velocity"].quantile(
    [0, 0.01, 0.25, 0.50, 0.75, 0.99, 1]
))

print("\nNON-POSITIVE VALUES:")
print("Diameter <= 0:", (df["Diameter"] <= 0).sum())
print("Velocity <= 0:", (df["Velocity"] <= 0).sum())

print("\nDUPLICATE BREAKDOWN:")
print("All columns:", df.duplicated().sum())
print("Diameter + Velocity:",
      df.duplicated(["Diameter", "Velocity"]).sum())
print("DateTime only:", df["DateTime"].duplicated().sum())

print("\nTIME RANGE:")
times = pd.to_datetime(df["DateTime"], errors="coerce")
print("Start:", times.min())
print("End:", times.max())

print("\nCHECK COMPLETE")
