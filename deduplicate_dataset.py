
from pathlib import Path
import duckdb

INPUT_CSV = Path("data/master_dataset_temp.csv")
OUTPUT_CSV = Path("data/raindrop_volume_deduplicated.csv")
DB_PATH = Path("data/duplicate_check_temp.duckdb")
TEMP_DIR = Path("data/duckdb_temp")

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

if OUTPUT_CSV.exists():
    raise FileExistsError(
        f"Output already exists; refusing to overwrite: {OUTPUT_CSV}"
    )

TEMP_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_PATH))

try:
    con.execute("SET memory_limit = '4GB'")
    con.execute("SET threads = 4")
    con.execute(
        f"SET temp_directory = '{TEMP_DIR.as_posix()}'"
    )

    print("Removing exact duplicate rows...")
    print("Original dataset:", INPUT_CSV)
    print("New dataset:", OUTPUT_CSV)

    con.execute("""
        COPY (
            SELECT DISTINCT *
            FROM read_csv(
                'data/master_dataset_temp.csv',
                header=true,
                delim=',',
                strict_mode=false,
                parallel=false
            )
        )
        TO 'data/raindrop_volume_deduplicated.csv'
        (HEADER, DELIMITER ',')
    """)

    result = con.execute("""
        SELECT COUNT(*)
        FROM read_csv(
            'data/raindrop_volume_deduplicated.csv',
            header=true,
            delim=',',
            strict_mode=false,
            parallel=false
        )
    """).fetchone()[0]

    print("Deduplication completed.")
    print("Rows in cleaned dataset:", result)
    print("Cleaned file size (bytes):", OUTPUT_CSV.stat().st_size)
    print("Original dataset preserved.")

finally:
    con.close()
