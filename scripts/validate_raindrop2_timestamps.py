from pathlib import Path
import pandas as pd

# Folder containing the 182 CSV files
DATA_DIR = Path("data/raindrop2_raw/raindrop2")

# Find all CSV files
csv_files = sorted(DATA_DIR.glob("*.csv"))

print(f"Total CSV files found: {len(csv_files)}")

# Store validation results
results = []

for file_path in csv_files:
    try:
        # Skip the first metadata line beginning with #Condition
        df = pd.read_csv(file_path, skiprows=1)

        # Combine date and time into one datetime column
        timestamps = pd.to_datetime(
            df["D-M-Y"].astype(str) + " " + df["Time"].astype(str),
            format="%Y/%m/%d %H:%M:%S",
            errors="coerce",
        )

        # Calculate time difference between consecutive rows
        gaps = timestamps.diff().dropna()

        # Count intervals that are not exactly one minute
        invalid_gaps = (gaps != pd.Timedelta(minutes=1)).sum()

        results.append({
            "file": file_path.name,
            "records": len(df),
            "invalid_timestamps": timestamps.isna().sum(),
            "invalid_minute_gaps": int(invalid_gaps),
        })

    except Exception as error:
        print(f"ERROR in {file_path.name}: {error}")

# Display summary
if results:
    report = pd.DataFrame(results)

    print("\n--- Timestamp Validation Summary ---")
    print(f"Files checked: {len(report)}")
    print(f"Total records: {report['records'].sum()}")
    print(
        "Invalid timestamps:",
        report["invalid_timestamps"].sum(),
    )
    print(
        "Invalid one-minute intervals:",
        report["invalid_minute_gaps"].sum(),
    )

    # Show files that need attention
    problems = report[
        (report["invalid_timestamps"] > 0)
        | (report["invalid_minute_gaps"] > 0)
    ]

    if problems.empty:
        print("\nAll checked timestamps are valid and continuous.")
    else:
        print("\nFiles requiring investigation:")
        print(problems.to_string(index=False))