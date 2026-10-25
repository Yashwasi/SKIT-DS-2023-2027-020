
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "models" / "clustered_raindrops_sample.csv"

df = pd.read_csv(DATA_PATH)

# Calculate volume from diameter
df["Calculated_Volume_mm3"] = (
    np.pi / 6 * df["Diameter_mm"] ** 3
)

df["Absolute_Difference"] = (
    df["Volume_mm3"] - df["Calculated_Volume_mm3"]
).abs()

print("TOTAL ROWS:", len(df))
print(
    "Rows matching the formula within 0.001 mm3:",
    (df["Absolute_Difference"] < 0.001).sum()
)
print(
    "Maximum absolute difference:",
    df["Absolute_Difference"].max()
)

print("\nLARGEST DIAMETERS AND VOLUMES:")
print(
    df.nlargest(10, "Diameter_mm")[
        [
            "Diameter_mm",
            "Volume_mm3",
            "Calculated_Volume_mm3",
            "Absolute_Difference",
            "Cluster"
        ]
    ].to_string(index=False)
)

