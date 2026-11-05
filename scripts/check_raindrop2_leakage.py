import pandas as pd
import zipfile
from io import StringIO

ZIP_PATH = "raindrop2.zip"

frames = []

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
        frames.append(df)

data = pd.concat(frames, ignore_index=True)

# Convert timestamp
data["Timestamp"] = pd.to_datetime(
    data["D-M-Y"].astype(str) + " " +
    data["Time"].astype(str),
    dayfirst=True,
    errors="coerce"
)

data["RainRate(mm/h)"] = pd.to_numeric(
    data["RainRate(mm/h)"],
    errors="coerce"
)

data = data.sort_values(["Condition", "Timestamp"])

# Next-minute target
data["Next_RainRate"] = (
    data.groupby("Condition")["RainRate(mm/h)"].shift(-1)
)

# Check consecutive one-minute intervals
data["Time_Gap"] = (
    data.groupby("Condition")["Timestamp"].diff()
)

valid_next = data[
    (data["Time_Gap"] == pd.Timedelta(minutes=1))
    & data["Next_RainRate"].notna()
]

print("\n========== LEAKAGE ANALYSIS ==========")

print("Total records:", len(data))

print("Valid consecutive minute pairs:", len(valid_next))

print(
    "Coverage (%):",
    round(len(valid_next) / len(data) * 100, 2)
)

print("\n========== SAME-MINUTE CORRELATION ==========")

features = [
    "Rain(mm)",
    "N",
    "KE(J/m2)",
    "nKE(J/m2/mm)",
    "D50",
    "Dmax",
    "N-1.0",
    "N-2.0",
    "R-1.0",
    "R-2.0"
]

for col in features:

    if col in data.columns:

        values = data[[col, "RainRate(mm/h)"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()

        corr = values.corr().iloc[0, 1]

        print(f"{col:18} : {corr:.4f}")

print("\n========== NEXT-MINUTE TARGET ==========")

print("Valid next-minute targets:", valid_next["Next_RainRate"].count())

print("\n========== COMPLETE ==========")