import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO

ZIP_PATH = "raindrop2.zip"
OUTPUT = "reports/raindrop2_analysis/plots"

os.makedirs(OUTPUT, exist_ok=True)

frames = []

with zipfile.ZipFile(ZIP_PATH) as z:

    files = [
        f for f in z.namelist()
        if f.lower().endswith(".csv")
    ]

    for file in files:

        content = z.read(file).decode("utf-8-sig")

        lines = [
            line for line in content.splitlines()
            if not line.startswith("#")
        ]

        df = pd.read_csv(StringIO("\n".join(lines)))
        frames.append(df)

data = pd.concat(frames, ignore_index=True)

selected = [
    "Rain(mm)",
    "RainRate(mm/h)",
    "KE(J/m2)",
    "nKE(J/m2/mm)",
    "N",
    "D50",
    "Dmax",
    "N-1.0",
    "N-2.0",
    "N-3.0",
    "R-1.0",
    "R-2.0",
    "R-3.0"
]

numeric = data[selected].apply(pd.to_numeric, errors="coerce")

correlation = numeric.corr()

plt.figure(figsize=(13, 10))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Rainfall Dataset - Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT}/06_correlation_heatmap.png",
    dpi=150
)

plt.close()

print("Heatmap generated successfully.")
print("Location:", f"{OUTPUT}/06_correlation_heatmap.png")