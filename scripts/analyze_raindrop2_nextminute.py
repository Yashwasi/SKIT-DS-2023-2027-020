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

        df = df.assign(Condition=condition)

        frames.append(df)

data = pd.concat(frames, ignore_index=True)

# Parse dates safely
date_text = data["D-M-Y"].astype(str).str.strip()
time_text = data["Time"].astype(str).str.strip()

date_a = pd.to_datetime(
    date_text + " " + time_text,
    format="%d-%m-%Y %H:%M:%S",
    errors="coerce"
)

date_b = pd.to_datetime(
    date_text + " " + time_text,
    format="%Y/%m/%d %H:%M:%S",
    errors="coerce"
)

data["Timestamp"] = date_a.fillna(date_b)

data["RainRate(mm/h)"] = pd.to_numeric(
    data["RainRate(mm/h)"],
    errors="coerce"
)

data = data.sort_values(["Condition", "Timestamp"])

# Create future target
data["Next_RainRate"] = (
    data.groupby("Condition")["RainRate(mm/h)"].shift(-1)
)

data["Next_Timestamp"] = (
    data.groupby("Condition")["Timestamp"].shift(-1)
)

# Keep only consecutive minutes
valid = data[
    (data["Next_Timestamp"] - data["Timestamp"])
    == pd.Timedelta(minutes=1)
].copy()

valid = valid.dropna(
    subset=["RainRate(mm/h)", "Next_RainRate"]
)

# Correlation
corr = valid[
    ["RainRate(mm/h)", "Next_RainRate"]
].corr().iloc[0, 1]

print("\n========== NEXT-MINUTE ANALYSIS ==========")

print("Valid pairs:", len(valid))

print("Current RainRate mean:",
      round(valid["RainRate(mm/h)"].mean(), 4))

print("Next RainRate mean:",
      round(valid["Next_RainRate"].mean(), 4))

print("Current vs Next correlation:",
      round(corr, 4))

print("\n========== COMPLETE ==========")

# Save validated pairs
valid.to_csv(
    "reports/raindrop2_analysis/next_minute_pairs.csv",
    index=False
)

print("Saved: next_minute_pairs.csv")
