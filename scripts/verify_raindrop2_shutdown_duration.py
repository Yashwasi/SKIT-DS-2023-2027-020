import zipfile
import pandas as pd
import io

ZIP_FILE = "raindrop2.zip"

shutdown_conditions = []

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

        rain = pd.to_numeric(
            df["RainRate(mm/h)"],
            errors="coerce"
        )

        time = pd.to_datetime(
            df["Time"].astype(str),
            format="%H:%M:%S",
            errors="coerce"
        )

        minutes = time.dt.strftime("%H:%M")

        at_0131 = rain[minutes == "01:31"]

        if len(at_0131) and at_0131.iloc[0] == 0:

            after = rain[
                (minutes >= "01:31") &
                (minutes <= "02:00")
            ]

            shutdown_conditions.append({
                "Condition": file.split("[")[-1].replace("].csv", ""),
                "Zero_Minutes": int((after == 0).sum()),
                "Total_Minutes": len(after),
                "All_Zero": bool((after == 0).all())
            })

result = pd.DataFrame(shutdown_conditions)

print("\n========== SHUTDOWN DURATION ==========")

print("Total shutdown conditions:", len(result))

print("Conditions with all 30 minutes zero:",
      result["All_Zero"].sum())

print("Average zero minutes:",
      round(result["Zero_Minutes"].mean(), 2))

print("\n========== COMPLETE ==========")