import zipfile
import pandas as pd
import io

ZIP_FILE = "raindrop2.zip"

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

        df["RainRate(mm/h)"] = pd.to_numeric(
            df["RainRate(mm/h)"],
            errors="coerce"
        )

        df["Condition"] = (
            file.split("[")[-1].replace("].csv", "")
        )

        all_data.append(
            df[["Time", "RainRate(mm/h)", "Condition"]]
        )

data = pd.concat(all_data, ignore_index=True)

data["Time"] = pd.to_datetime(
    data["Time"].astype(str),
    format="%H:%M:%S",
    errors="coerce"
)

data = data.dropna(subset=["Time"])

data["Minute"] = data["Time"].dt.strftime("%H:%M")

results = []

for condition, group in data.groupby("Condition"):

    group = group.sort_values("Time")

    at_0131 = group[
        group["Minute"] == "01:31"
    ]

    if at_0131.empty:
        continue

    if at_0131["RainRate(mm/h)"].iloc[0] == 0:

        after = group[
            group["Minute"] > "01:31"
        ]

        restarted = (
            after["RainRate(mm/h)"] > 0
        ).any()

        results.append({
            "Condition": condition,
            "Rainfall_Restarted": restarted
        })

result = pd.DataFrame(results)

print("\n========== RAINFALL RESTART ANALYSIS ==========")

print("Shutdown conditions:", len(result))

print(
    "Conditions with rainfall restarting:",
    result["Rainfall_Restarted"].sum()
)

print(
    "Conditions remaining zero:",
    (~result["Rainfall_Restarted"]).sum()
)

print("\n========== COMPLETE ==========")