import zipfile
import csv
from collections import Counter

archive = zipfile.ZipFile("raindrop2.zip", "r")
files = [f for f in archive.namelist() if f.lower().endswith(".csv")]

row_counts = []
schemas = set()
bad_files = []

for filename in files:
    content = archive.read(filename).decode("utf-8-sig")
    lines = content.splitlines()

    header_index = next(
        (i for i, line in enumerate(lines) if not line.startswith("#")),
        None
    )

    if header_index is None:
        bad_files.append(filename)
        continue

    reader = csv.reader(lines[header_index:])
    columns = next(reader)
    schemas.add(tuple(columns))

    row_counts.append(sum(1 for row in reader))

archive.close()

print("\n========== DATASET INSPECTION ==========")
print("Total CSV files:", len(files))
print("Total data rows:", sum(row_counts))
print("Rows per file:", dict(Counter(row_counts)))
print("Unique schemas:", len(schemas))
print("Number of columns:", len(next(iter(schemas))) if schemas else 0)
print("Files without headers:", bad_files)
print("========================================")
