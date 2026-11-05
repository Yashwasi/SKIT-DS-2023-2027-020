import zipfile
import csv
from io import StringIO
from collections import Counter

ZIP_PATH = "raindrop2.zip"

conditions = Counter()

with zipfile.ZipFile(ZIP_PATH) as z:

    files = [
        f for f in z.namelist()
        if f.lower().endswith(".csv")
    ]

    for file in files:

        content = z.read(file).decode("utf-8-sig")
        lines = content.splitlines()

        metadata = next(
            (line for line in lines if line.startswith("#Condition:")),
            None
        )

        condition = (
            metadata.replace("#Condition:", "").strip()
            if metadata else "Unknown"
        )

        csv_lines = [
            line for line in lines
            if not line.startswith("#")
        ]

        reader = csv.DictReader(StringIO("\n".join(csv_lines)))

        rows = sum(1 for _ in reader)

        conditions[condition] += rows

print("\n========== CONDITION REPORT ==========")

print("Unique conditions:", len(conditions))

print("\nFirst 20 conditions:")

for condition, count in list(conditions.items())[:20]:
    print(f"{condition}: {count} records")

print("\n========== SUMMARY ==========")

print("Total condition records:", sum(conditions.values()))
print("Minimum records:", min(conditions.values()))
print("Maximum records:", max(conditions.values()))

print("\n========== COMPLETE ==========")