import zipfile
import duckdb
import csv
import io

zip_path = "raindrop2.zip"
con = duckdb.connect()

frames = []

with zipfile.ZipFile(zip_path) as archive:
    files = [f for f in archive.namelist() if f.endswith(".csv")]

    for filename in files:
        content = archive.read(filename).decode("utf-8-sig")
        lines = content.splitlines()

        header_index = next(
            (i for i, line in enumerate(lines) if not line.startswith("#")),
            None
        )

        if header_index is None:
            continue

        metadata = lines[0].replace("#Condition:", "").strip()
        csv_text = "\n".join(lines[header_index:])

        table = con.execute(
            "SELECT * FROM read_csv(?, header=true, auto_detect=true)",
            [csv_text]
        ).df()

        table["Condition_File"] = metadata
        frames.append(table)

data = con.execute(
    "SELECT * FROM frames"
).df()

print("\n========== DATA QUALITY REPORT ==========")
print("Total rows:", len(data))
print("Total columns:", len(data.columns))

print("\nMissing values:")
for column in data.columns:
    count = int(data[column].isna().sum())
    if count > 0:
        print(column, ":", count)

print("\nExact duplicate rows:", int(data.duplicated().sum()))

print("\nRainRate Summary:")
print(data["RainRate(mm/h)"].describe())

print("\nNegative RainRate values:",
      int((data["RainRate(mm/h)"] < 0).sum()))

print("\nZero RainRate values:",
      int((data["RainRate(mm/h)"] == 0).sum()))

print("\nUnique conditions:", data["Condition_File"].nunique())

print("=========================================")

con.close()
