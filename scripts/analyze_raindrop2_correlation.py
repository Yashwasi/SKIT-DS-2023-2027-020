import pandas as pd
import zipfile
import os
from io import StringIO

ZIP_PATH = "raindrop2.zip"
OUTPUT_DIR = "reports/raindrop2_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataframes = []

with zipfile.ZipFile(ZIP_PATH) as z:

    files = [
        f for f in z.namelist()
        if f.lower().endswith(".csv")
    ]

    for file in files:

        content = z.read(file).decode("utf-8-sig")

        condition = file.split("[")[-1].replace("].csv", "")

        lines = [
            line for line in content.splitlines()
            if not line.startswith("#")
        ]

        df = pd.read_csv(StringIO("\n".join(lines)))
        df["Condition"] = condition

        dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

# Numeric columns only
numeric_data = data.select_dtypes(include="number")

# Correlation with target
correlation = numeric_data.corr()["RainRate(mm/h)"]

correlation = correlation.drop("RainRate(mm/h)")
correlation = correlation.sort_values(
    key=abs,
    ascending=False
)

# Save complete correlation
correlation.to_csv(
    f"{OUTPUT_DIR}/rainrate_correlation.csv",
    header=["Correlation"]
)

print("\n========== TOP 15 CORRELATED FEATURES ==========")

print(correlation.head(15).round(4).to_string())

print("\n========== BOTTOM 10 CORRELATED FEATURES ==========")

print(correlation.tail(10).round(4).to_string())

print("\nCorrelation report saved successfully.")