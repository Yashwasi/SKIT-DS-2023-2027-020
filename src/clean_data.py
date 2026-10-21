
import zipfile
from pathlib import Path

import pandas as pd

ZIP_PATH = Path(
    r"C:\Users\shubh\Downloads\raindrop_volume_processed.zip"
)

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "raindrop_volume_working.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"{OUTPUT_FILE} already exists. Rename it or choose another output."
    )

with zipfile.ZipFile(ZIP_PATH) as archive:
    with archive.open("raindrop_volume_dataset.csv") as source:
        with OUTPUT_FILE.open("wb") as destination:
            while True:
                block = source.read(8 * 1024 * 1024)
                if not block:
                    break
                destination.write(block)

print("Working dataset created:", OUTPUT_FILE)
print("Original ZIP was not modified.")
print("No rows were deleted or changed.")
