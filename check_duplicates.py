
import zipfile
from pathlib import Path

import duckdb

ZIP_PATH = Path(
    r"C:\Users\shubh\Downloads\raindrop_volume_processed.zip"
)
TEMP_CSV = Path("data") / "master_dataset_temp.csv"
DB_PATH = Path("data") / "duplicate_check_temp.duckdb"
TEMP_DIR = Path("data") / "duckdb_temp"

TEMP_CSV.parent.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"ZIP file not found: {ZIP_PATH}")

if not TEMP_CSV.exists():
    raise FileNotFoundError(
        f"Existing CSV not found: {TEMP_CSV}"
    )
if DB_PATH.exists():
    print("Using existing temporary database:", DB_PATH)

con = None

try:


    con = duckdb.connect(str(DB_PATH))
    con.execute("SET memory_limit = '4GB'")
    con.execute("SET threads = 4")
    con.execute(
        "SET temp_directory = 'data/duckdb_temp'"
    )

    total_rows = con.execute(
    "SELECT COUNT(*) FROM read_csv(?, header=true, delim=',', strict_mode=false, parallel=false)",
    [str(TEMP_CSV)]
).fetchone()[0]

    duplicate_rows = con.execute("""
        SELECT COALESCE(SUM(n - 1), 0)
        FROM (
            SELECT COUNT(*) AS n
            FROM read_csv(
                ?,
                header=true,
                delim=',',
                strict_mode=false,
                parallel=false
            )
            GROUP BY
                DateTime,
                Diameter_mm,
                Velocity_m_s,
                Volume_mm3,
                Condition,
                SourceFile
            HAVING COUNT(*) > 1
        )
    """, [str(TEMP_CSV)]).fetchone()[0]
    print("Total rows:", total_rows)
    print("Full-row duplicate rows:", duplicate_rows)
    print("Rows after removing exact duplicates:", total_rows - duplicate_rows)
except Exception as error:
    print("Duplicate check failed:", error)
    print("Temporary files have been preserved for inspection.")
    raise

finally:
    if con is not None:
        con.close()

print("Duplicate check completed.")
print("Temporary CSV and database preserved for inspection.")
