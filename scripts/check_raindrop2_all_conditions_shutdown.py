import zipfile
import pandas as pd
import io

ZIP_FILE = "raindrop2.zip"

results = []

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

        df["Time"] = pd.to_datetime(
            df["Time"].astype(str),
            format="%H:%M:%S",
            errors="coerce"
        ).dt.strftime("%H:%M")

        before = df[
            df["Time"] == "01:30"
        ]["RainRate(mm/h)"]

        after = df[
            df["Time"] == "01:31"
        ]["RainRate(mm/h)"]

        if len(before) and len(after):

            results.append({
                "Condition": file.split("[")[-1].replace("].csv", ""),
                "Rain_0130": before.iloc[0],
                "Rain_0131": after.iloc[0],
                "Shutdown": after.iloc[0] == 0
            })

result = pd.DataFrame(results)

print("\n========== ALL CONDITIONS ANALYSIS ==========")

print("Total conditions:", len(result))
print("Conditions with zero at 01:31:",
      result["Shutdown"].sum())

print("Shutdown percentage:",
      round(result["Shutdown"].mean() * 100, 2), "%")

print("\nConditions with non-zero rainfall at 01:31:")

print(
    result[result["Rain_0131"] != 0].to_string(index=False)
)

result.to_csv(
    "reports/raindrop2_analysis/all_conditions_shutdown.csv",
    index=False
)

print("\nReport saved.")