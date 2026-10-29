
from pathlib import Path
import pandas as pd
import numpy as np

folder = Path.home() / "Downloads" / "raindrop_dataset"
files = sorted(folder.rglob("*.csv"))

if not files:
    raise FileNotFoundError(f"No CSV files found in {folder}")

file = files[0]

# Read a small sample only
df = pd.read_csv(file, comment="#", nrows=10_000)

# Calculate volume assuming spherical drops.
# This assumes Diameter is measured in millimetres.
df["Volume_mm3"] = (np.pi / 6) * df["Diameter"] ** 3

print("SOURCE FILE:", file.name)
print("\nSAMPLE WITH CALCULATED VOLUME:")
print(df.head(10).to_string(index=False))

print("\nVOLUME SUMMARY (mm^3):")
print(df["Volume_mm3"].describe())

output = Path("volume_sample.csv")
df.to_csv(output, index=False)

print("\nSample saved to:", output.resolve())
