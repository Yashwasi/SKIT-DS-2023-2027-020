import zipfile
import csv
from io import StringIO
from collections import Counter
from datetime import datetime

ZIP_PATH = "raindrop2.zip"

all_timestamps = Counter()
date_counts = Counter()
invalid_dates = []

def parse_timestamp(date, time):
    formats = [
        "%d-%m-%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S"
    ]

    value = date.strip() + " " + time.strip()

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return None


with zipfile.ZipFile(ZIP_PATH) as z:

    files = [
        f for f in z.namelist()
        if f.lower().endswith(".csv")
    ]

    for file in files:

        content = z.read(file).decode("utf-8-sig")

        lines = [
            line for line in content.splitlines()
            if not line.startswith("#")
        ]

        reader = csv.DictReader(StringIO("\n".join(lines)))

        for row in reader:

            dt = parse_timestamp(
                row["D-M-Y"],
                row["Time"]
            )

            if dt is None:
                invalid_dates.append(row["D-M-Y"])
                continue

            all_timestamps[dt] += 1
            date_counts[dt.date()] += 1


print("\n========== TIMESTAMP REPORT ==========")

timestamps = sorted(all_timestamps)

print("Unique timestamps :", len(timestamps))
print("First timestamp   :", timestamps[0] if timestamps else "N/A")
print("Last timestamp    :", timestamps[-1] if timestamps else "N/A")
print("Unique dates      :", len(date_counts))
print("Invalid dates     :", len(invalid_dates))

print("\n========== RECORDS PER DATE ==========")

for date, count in sorted(date_counts.items()):
    print(date, ":", count)

print("\n========== SAMPLING ANALYSIS ==========")

if all_timestamps:
    print("Minimum records per timestamp :", min(all_timestamps.values()))
    print("Maximum records per timestamp :", max(all_timestamps.values()))

    print("Average records per timestamp :", round(
        sum(all_timestamps.values()) / len(all_timestamps), 2
    ))

print("\n========== ANALYSIS COMPLETE ==========")