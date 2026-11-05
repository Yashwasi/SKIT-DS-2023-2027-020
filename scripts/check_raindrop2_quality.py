import zipfile
import csv
import io
import math
from collections import Counter

ZIP_PATH = "raindrop2.zip"

total_rows = 0
total_missing = 0
total_duplicates = 0
invalid_timestamps = 0
rain_values = []
column_counts = Counter()

with zipfile.ZipFile(ZIP_PATH) as z:

    files = [
        f for f in z.namelist()
        if f.lower().endswith(".csv")
    ]

    for file in files:

        content = z.read(file).decode("utf-8-sig")
        lines = content.splitlines()

        csv_lines = [
            line for line in lines
            if not line.startswith("#")
        ]

        reader = csv.DictReader(csv_lines)

        seen = set()

        for row in reader:
            total_rows += 1

            values = tuple(row.values())

            if values in seen:
                total_duplicates += 1
            else:
                seen.add(values)

            for col, value in row.items():
                column_counts[col] += 1

                if value is None or value.strip() == "":
                    total_missing += 1

            rain = row.get("RainRate(mm/h)", "")

            if rain.strip():
                try:
                    value = float(rain)

                    if math.isfinite(value):
                        rain_values.append(value)

                except ValueError:
                    pass

print("\n========== DATA QUALITY REPORT ==========")
print("Total CSV files      :", len(files))
print("Total records        :", total_rows)
print("Missing cells        :", total_missing)
print("Duplicate rows       :", total_duplicates)

print("\n========== RAIN RATE SUMMARY ==========")

rain_values.sort()

if rain_values:
    n = len(rain_values)

    print("Valid observations :", n)
    print("Minimum            :", min(rain_values))
    print("Maximum            :", max(rain_values))
    print("Mean               :", sum(rain_values) / n)
    print("Median             :", (
        rain_values[n // 2] if n % 2 else
        (rain_values[n // 2 - 1] + rain_values[n // 2]) / 2
    ))
    print("Zero rainfall (%)  :", sum(v == 0 for v in rain_values) / n * 100)

print("\n========== ANALYSIS COMPLETE ==========")