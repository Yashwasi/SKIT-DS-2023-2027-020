
from pathlib import Path
import shutil

SOURCE_FILE = Path("data/raindrop_volume_deduplicated.csv")
OUTPUT_FILE = Path("data/raindrop_volume_working.csv")

if not SOURCE_FILE.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {SOURCE_FILE}"
    )

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"{OUTPUT_FILE} already exists. "
        "Rename it or choose another output."
    )

shutil.copy2(SOURCE_FILE, OUTPUT_FILE)

print("Working dataset created:", OUTPUT_FILE)
print("Source: deduplicated dataset")
print("Rows copied without further changes.")
