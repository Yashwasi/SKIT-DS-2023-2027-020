import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import StringIO

ZIP_PATH = "raindrop2.zip"
OUTPUT_DIR = "reports/raindrop2_analysis/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataframes = []

print("Reading ZIP files...")

with zipfile.ZipFile(ZIP_PATH) as z:

    files = [
        f for f in z.namelist()
        if f.lower().endswith(".csv")
    ]

    for i, file in enumerate(files, 1):

        content = z.read(file).decode("utf-8-sig")

        condition = file.split("[")[-1].replace("].csv", "")

        lines = [
            line for line in content.splitlines()
            if not line.startswith("#")
        ]

        df = pd.read_csv(StringIO("\n".join(lines)))

        df["Condition"] = condition

        dataframes.append(df)

        if i % 50 == 0:
            print(f"Processed {i}/{len(files)} files")

data = pd.concat(dataframes, ignore_index=True)

print("\nDataset loaded:", data.shape)

# Convert numerical columns
numeric_columns = [
    "RainRate(mm/h)",
    "Rain(mm)",
    "KE(J/m2)",
    "nKE(J/m2/mm)",
    "D50"
]

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# 1. RainRate Distribution
plt.figure(figsize=(9, 5))
plt.hist(data["RainRate(mm/h)"].dropna(), bins=50)
plt.title("RainRate Distribution")
plt.xlabel("RainRate (mm/h)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_rainrate_distribution.png", dpi=150)
plt.close()

# 2. RainRate Boxplot
plt.figure(figsize=(9, 5))
plt.boxplot(data["RainRate(mm/h)"].dropna())
plt.title("RainRate Boxplot")
plt.ylabel("RainRate (mm/h)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_rainrate_boxplot.png", dpi=150)
plt.close()

# 3. RainRate vs Rain
plt.figure(figsize=(8, 5))
plt.scatter(
    data["Rain(mm)"],
    data["RainRate(mm/h)"],
    alpha=0.3
)
plt.title("Rain vs RainRate")
plt.xlabel("Rain (mm)")
plt.ylabel("RainRate (mm/h)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_rain_vs_rainrate.png", dpi=150)
plt.close()

# 4. RainRate vs KE
plt.figure(figsize=(8, 5))
plt.scatter(
    data["KE(J/m2)"],
    data["RainRate(mm/h)"],
    alpha=0.3
)
plt.title("Kinetic Energy vs RainRate")
plt.xlabel("KE (J/m2)")
plt.ylabel("RainRate (mm/h)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_ke_vs_rainrate.png", dpi=150)
plt.close()

# 5. RainRate by Condition (top 15)
top_conditions = (
    data.groupby("Condition")["RainRate(mm/h)"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(12, 6))
top_conditions.sort_values().plot(kind="barh")
plt.title("Top 15 Conditions by Average RainRate")
plt.xlabel("Average RainRate (mm/h)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_condition_comparison.png", dpi=150)
plt.close()

# Save summary
summary = data[numeric_columns].describe()
summary.to_csv("reports/raindrop2_analysis/numeric_summary.csv")

print("\n========== EDA COMPLETE ==========")
print("Total records:", len(data))
print("Total features:", len(data.columns))
print("Plots generated: 5")
print("Output folder:", OUTPUT_DIR)
print("Summary saved: reports/raindrop2_analysis/numeric_summary.csv")