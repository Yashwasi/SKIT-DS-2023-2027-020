
import zipfile
import pandas as pd
from collections import Counter

zip_path = r"C:\Users\shubh\Downloads\raindrop_volume_processed.zip"

counts = Counter()
total_rows = 0

with zipfile.ZipFile(zip_path) as z:
    with z.open("raindrop_volume_dataset.csv") as f:
        for chunk in pd.read_csv(
            f,
            usecols=["SourceFile"],
            chunksize=200_000
        ):
            total_rows += len(chunk)
            counts.update(chunk["SourceFile"].value_counts().to_dict())

print("Total rows:", total_rows)
print("Number of source files:", len(counts))
print("\nRows per source file:")

for filename, count in sorted(counts.items()):
    print(f"{filename}: {count}")
