import zipfile
import csv
from io import StringIO

ZIP_PATH = "raindrop2.zip"

with zipfile.ZipFile(ZIP_PATH) as z:

    files = [
        f for f in z.namelist()
        if f.lower().endswith(".csv")
    ]

    content = z.read(files[0]).decode("utf-8-sig")

    lines = [
        line for line in content.splitlines()
        if not line.startswith("#")
    ]

    reader = csv.DictReader(StringIO("\n".join(lines)))

    columns = reader.fieldnames

    print("\n========== FEATURE REPORT ==========")

    print("Total features:", len(columns))

    print("\nAll available features:\n")

    for i, column in enumerate(columns, 1):
        print(f"{i}. {column}")

    print("\n========== TARGET CHECK ==========")

    targets = [
        "RainRate(mm/h)",
        "Rain(mm)",
        "N",
        "KE",
        "nKE"
    ]

    for target in targets:
        print(
            target,
            "->",
            "AVAILABLE" if target in columns else "NOT AVAILABLE"
        )

    print("\n========== COMPLETE ==========")