
from pathlib import Path
from datetime import datetime
import duckdb

DATASET = Path("data/raindrop_volume_deduplicated.csv")
REPORT_DIR = Path("reports")
REPORT_FILE = REPORT_DIR / "dataset_validation_report.txt"

if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset nahi mili: {DATASET.resolve()}"
    )

con = duckdb.connect()
con.execute("SET memory_limit = '4GB'")
con.execute("SET threads = 4")

csv_path = DATASET.resolve().as_posix().replace("'", "''")
source = f"""
    read_csv(
        '{csv_path}',
        header=true,
        delim=',',
        strict_mode=false,
        parallel=false
    )
"""

summary = con.execute(f"""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT Condition) AS unique_conditions,
        COUNT(DISTINCT SourceFile) AS unique_source_files,
        COUNT(*) - COUNT(DISTINCT (
            DateTime, Diameter_mm, Velocity_m_s,
            Volume_mm3, Condition, SourceFile
        )) AS duplicate_rows
    FROM {source}
""").fetchone()

missing = con.execute(f"""
    SELECT
        COUNT(*) FILTER (WHERE DateTime IS NULL),
        COUNT(*) FILTER (WHERE Diameter_mm IS NULL),
        COUNT(*) FILTER (WHERE Velocity_m_s IS NULL),
        COUNT(*) FILTER (WHERE Volume_mm3 IS NULL),
        COUNT(*) FILTER (WHERE Condition IS NULL),
        COUNT(*) FILTER (WHERE SourceFile IS NULL)
    FROM {source}
""").fetchone()

conditions = con.execute(f"""
    SELECT Condition, COUNT(*) AS row_count
    FROM {source}
    GROUP BY Condition
    ORDER BY row_count DESC
""").fetchall()

con.close()

REPORT_DIR.mkdir(exist_ok=True)

lines = [
    "RAINDROP VOLUME DATASET - VALIDATION REPORT",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "DATASET SUMMARY",
    f"Total rows: {summary[0]:,}",
    f"Unique conditions: {summary[1]}",
    f"Unique source files: {summary[2]}",
    f"Exact duplicate rows remaining: {summary[3]:,}",
    "",
    "MISSING VALUES",
]

column_names = [
    "DateTime", "Diameter_mm", "Velocity_m_s",
    "Volume_mm3", "Condition", "SourceFile"
]

for name, count in zip(column_names, missing):
    lines.append(f"{name}: {count:,}")

lines += ["", "RECORDS PER CONDITION"]

for condition, count in conditions:
    lines.append(f"{condition}: {count:,}")

lines += [
    "",
    "NOTE",
    "Earlier checks found no invalid measurements and no formula",
    "mismatches for Volume_mm3 = (pi/6) * Diameter_mm^3.",
    "Those checks are recorded separately; this script does not",
    "rerun the formula or invalid-measurement tests.",
]

REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

print("Report saved:", REPORT_FILE.resolve())
print("Rows:", f"{summary[0]:,}")
print("Conditions:", summary[1])
print("Exact duplicate rows remaining:", f"{summary[3]:,}")
