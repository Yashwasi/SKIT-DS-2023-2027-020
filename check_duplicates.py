
import duckdb

zip_path = r"C:\Users\shubh\Downloads\raindrop_volume_processed.zip"

# Extract the CSV to a temporary file only if needed.
import zipfile
from pathlib import Path

temp_csv = Path("data") / "master_dataset_temp.csv"
temp_csv.parent.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path) as archive:
    with archive.open("raindrop_volume_dataset.csv") as source:
        with open(temp_csv, "wb") as destination:
            while True:
                block = source.read(8 * 1024 * 1024)
                if not block:
                    break
                destination.write(block)

con = duckdb.connect()

result = con.execute("""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(*) - COUNT(DISTINCT hash(
            DateTime, Diameter_mm, Velocity_m_s,
            Volume_mm3, Condition, SourceFile
        )) AS duplicate_candidates
FROM read_csv(?, delim=',', header=true, strict_mode=false, parallel=false)""", [str(temp_csv)]).fetchone()

print("Total rows:", result[0])
print("Duplicate candidates:", result[1])

con.close()
temp_csv.unlink(missing_ok=True)
