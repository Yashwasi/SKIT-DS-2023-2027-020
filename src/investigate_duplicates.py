"""
Faster duplicate investigation for the raindrop dataset.
Run from project root:
    python src/investigate_duplicates_fast.py

Reads the ZIP in chunks and uses vectorized pandas hashes.
Does not change or delete the original dataset.
It checks exact duplicates within chunks and uses a disk-backed SQLite table
to identify fingerprints recurring across chunks, with batched inserts.
"""
from pathlib import Path
import sqlite3
import zipfile
import pandas as pd

ZIP_PATH = Path("data/raindrop_volume_processed.zip")
OUT_DIR = Path("models")
CHUNK_SIZE = 200_000
DB_PATH = OUT_DIR / "duplicate_fingerprints_fast.sqlite"
SUMMARY_PATH = OUT_DIR / "duplicate_investigation_summary_fast.csv"

def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Missing {ZIP_PATH}; run from project root.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise RuntimeError(f"Expected one CSV in ZIP, found {len(csvs)}.")
        csv_name = csvs[0]
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity error: {bad}")

    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("""CREATE TABLE fp (
        h TEXT PRIMARY KEY,
        n INTEGER NOT NULL,
        first_chunk INTEGER NOT NULL,
        chunks_seen INTEGER NOT NULL
    )""")
    conn.commit()

    rows_total = 0
    within_chunk_redundant = 0
    chunk_no = 0
    cross_chunk_new_occurrences = 0
    try:
        with zipfile.ZipFile(ZIP_PATH) as zf, zf.open(csv_name) as raw:
            for chunk in pd.read_csv(raw, chunksize=CHUNK_SIZE):
                chunk_no += 1
                n = len(chunk)
                rows_total += n
                # Hash rows as uint64; vectorized value_counts is much faster than
                # Python-level per-row processing.
                hashes = pd.util.hash_pandas_object(chunk, index=False).astype("uint64")
                counts = hashes.value_counts(sort=False)
                within_chunk_redundant += int((counts - 1).clip(lower=0).sum())

                # One SQLite lookup per unique hash in this chunk, batched via temp table.
                conn.execute("DROP TABLE IF EXISTS temp.current_chunk")
                conn.execute("CREATE TEMP TABLE current_chunk (h TEXT PRIMARY KEY, n INTEGER)")
                conn.executemany(
                    "INSERT INTO current_chunk(h,n) VALUES (?,?)",
                    ((str(int(h)), int(c)) for h, c in counts.items())
                )

                # Fingerprints already seen in a previous chunk; count current
                # chunk occurrences that match earlier chunks.
                cross_chunk_new_occurrences += conn.execute("""
                    SELECT COALESCE(SUM(c.n),0)
                    FROM current_chunk c JOIN fp f ON c.h=f.h
                    WHERE f.first_chunk < ?
                """, (chunk_no,)).fetchone()[0]

                conn.execute("""
                    INSERT INTO fp(h,n,first_chunk,chunks_seen)
                    SELECT h,n,?,1 FROM current_chunk WHERE 1
                    ON CONFLICT(h) DO UPDATE SET
                        n=fp.n+excluded.n,
                        chunks_seen=fp.chunks_seen+1
                """, (chunk_no,))
                conn.commit()
                print(f"Processed chunk {chunk_no} | Rows checked: {rows_total:,}", flush=True)

        unique_fingerprints = conn.execute("SELECT COUNT(*) FROM fp").fetchone()[0]
        duplicate_patterns = conn.execute("SELECT COUNT(*) FROM fp WHERE n > 1").fetchone()[0]
        redundant_occurrences = conn.execute(
            "SELECT COALESCE(SUM(n-1),0) FROM fp"
        ).fetchone()[0]
        patterns_spanning_chunks = conn.execute(
            "SELECT COUNT(*) FROM fp WHERE chunks_seen > 1"
        ).fetchone()[0]

        summary = pd.DataFrame([
            ("Total rows scanned", rows_total),
            ("Unique row fingerprints", unique_fingerprints),
            ("Duplicate row patterns", duplicate_patterns),
            ("Redundant duplicate occurrences across all chunks", redundant_occurrences),
            ("Redundant occurrences detected within chunks", within_chunk_redundant),
            ("Rows in a chunk matching fingerprints from earlier chunks", cross_chunk_new_occurrences),
            ("Fingerprint patterns appearing in multiple chunks", patterns_spanning_chunks),
            ("Chunk size", CHUNK_SIZE),
            ("Method note", "64-bit pandas row fingerprints; no source rows deleted"),
        ], columns=["Check", "Result"])
        summary.to_csv(SUMMARY_PATH, index=False)
        print("\n===== DUPLICATE INVESTIGATION SUMMARY =====")
        print(summary.to_string(index=False))
        print(f"\nSaved: {SUMMARY_PATH}")
        print("\nNo source rows were deleted or changed.")
        print("Note: fingerprint matching is hash-based; collisions are extremely unlikely.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
