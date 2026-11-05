
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dataset location
folder = Path.home() / "Downloads" / "raindrop_dataset"
files = sorted(folder.rglob("*.csv"))

if not files:
    raise FileNotFoundError(
        f"No CSV files found in: {folder}"
    )

# Read a reproducible sample from the first CSV
df = pd.read_csv(
    files[0],
    comment="#",
    nrows=50_000
)

# Validate required columns
required = {"DateTime", "Diameter", "Velocity"}
missing_columns = required - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Convert measurements to numeric safely
df["Diameter"] = pd.to_numeric(
    df["Diameter"], errors="coerce"
)
df["Velocity"] = pd.to_numeric(
    df["Velocity"], errors="coerce"
)
df["DateTime"] = pd.to_datetime(
    df["DateTime"], errors="coerce"
)

# Report data quality before filtering
print("SOURCE FILE:", files[0].name)
print("SAMPLE ROWS:", len(df))
print("\nMISSING VALUES:")
print(df.isna().sum())

print("\nDUPLICATE ROWS IN SAMPLE:", df.duplicated().sum())

# Keep valid, positive diameter and velocity values
valid = (
    df["Diameter"].gt(0)
    & df["Velocity"].gt(0)
    & np.isfinite(df["Diameter"])
    & np.isfinite(df["Velocity"])
)

clean = df.loc[valid].copy()

# Volume assumes spherical drops.
# Units depend on the verified unit of Diameter.
clean["Volume_calculated"] = (
    np.pi / 6 * clean["Diameter"] ** 3
)

print("\nVALID ROWS:", len(clean))
print("\nMEASUREMENT SUMMARY:")
print(clean[["Diameter", "Velocity",
             "Volume_calculated"]].describe())

# Save numerical summary
summary_path = Path("eda_summary.csv")
clean[["Diameter", "Velocity",
       "Volume_calculated"]].describe().to_csv(summary_path)

# Graph 1: Diameter distribution
plt.figure(figsize=(8, 5))
plt.hist(clean["Diameter"], bins=40)
plt.xlabel("Diameter (unit to be verified)")
plt.ylabel("Number of observations")
plt.title("Distribution of Raindrop Diameter")
plt.tight_layout()
plt.savefig("diameter_distribution.png", dpi=150)
plt.close()

# Graph 2: Velocity vs diameter
plt.figure(figsize=(8, 5))
plt.scatter(
    clean["Diameter"],
    clean["Velocity"],
    s=5,
    alpha=0.25
)
plt.xlabel("Diameter (unit to be verified)")
plt.ylabel("Velocity (unit to be verified)")
plt.title("Raindrop Diameter vs Velocity")
plt.tight_layout()
plt.savefig("diameter_velocity.png", dpi=150)
plt.close()

# Graph 3: Volume distribution
volumes = clean["Volume_calculated"]
volumes = volumes[volumes > 0]

plt.figure(figsize=(8, 5))
plt.hist(volumes, bins=40)
plt.xscale("log")
plt.xlabel("Calculated volume (log scale; units to verify)")
plt.ylabel("Number of observations")
plt.title("Distribution of Calculated Raindrop Volume")
plt.tight_layout()
plt.savefig("volume_distribution.png", dpi=150)
plt.close()

print("\nSaved files:")
print("- diameter_distribution.png")
print("- diameter_velocity.png")
print("- volume_distribution.png")
print("- eda_summary.csv")
