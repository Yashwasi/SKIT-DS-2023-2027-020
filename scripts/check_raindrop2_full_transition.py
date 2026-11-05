import zipfile
import pandas as pd
import io

ZIP_FILE = "raindrop2.zip"

dates = [
    "2016-06-16",
    "2016-06-17",
    "2016-06-24",
    "2016-06-28"
]

all_data = []

with zipfile.ZipFile(ZIP_FILE, "r") as z:

    files = [
        f for f in z.namelist()
        if f.endswith(".csv")
    ]

    for file in files:

        with z.open(file) as f:

            lines = f.readlines()

        content = b"".join(
            line for line in lines
            if not line.startswith(b"#")
        )

        df = pd.read_csv(io.BytesIO(content))

        df["Condition"] = file.split("[")[-1].replace("].csv", "")

        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

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

data = data.dropna(
    subset=["Timestamp", "RainRate(mm/h)"]
)

for date in dates:

    subset = data[
        (data["Timestamp"].dt.strftime("%Y-%m-%d") == date) &
        (data["Timestamp"].dt.strftime("%H:%M") >= "01:20") &
        (data["Timestamp"].dt.strftime("%H:%M") <= "01:40")
    ]

    summary = subset.groupby(
        subset["Timestamp"].dt.strftime("%H:%M")
    )["RainRate(mm/h)"].agg(
        ["count", "mean", "min", "max"]
    )

    print(f"\n========== {date} ==========")
    print(summary.to_string())

print("\nFull dataset transition analysis complete.")