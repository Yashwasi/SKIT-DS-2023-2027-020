
from pathlib import Path
import duckdb
import math
import statistics
from datetime import datetime

# =====================================================
# PROJECT PATHS
# =====================================================

ROOT = Path(__file__).resolve().parent

DATASET = (
    ROOT
    / "raindrop_volume_processed"
    / "raindrop_volume_deduplicated.csv"
)

OUTPUT_DIR = ROOT / "reports" / "eda"
REPORT_FILE = OUTPUT_DIR / "EDA_Report.md"

# =====================================================
# VALIDATE FILES
# =====================================================

if not DATASET.exists():
    raise FileNotFoundError(
        f"Deduplicated dataset not found:\n{DATASET}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("RAIN DROP VOLUME PREDICTION USING AI")
print("Exploratory Data Analysis Report Generator")
print("=" * 60)

print("\nDataset:", DATASET)
print("Report:", REPORT_FILE)
print("\nAnalyzing dataset. Please wait...")

# =====================================================
# DUCKDB CONNECTION
# =====================================================

con = duckdb.connect(":memory:")

con.execute("SET threads = 2")
con.execute("SET memory_limit = '4GB'")

# CSV path is passed as a parameter, not embedded in SQL.
csv_path = str(DATASET).replace("\\", "/")

con.execute(
    f"""
    CREATE VIEW rain_data AS
    SELECT *
    FROM read_csv_auto(
        '{csv_path}',
        sample_size = 100000,
        strict_mode = false
    )
    """
)

# =====================================================
# BASIC DATASET INFORMATION
# =====================================================

row_count = con.execute(
    "SELECT COUNT(*) FROM rain_data"
).fetchone()[0]

columns = con.execute(
    "DESCRIBE rain_data"
).fetchall()

column_names = [column[0] for column in columns]

print(f"\nTotal records: {row_count:,}")
print("Columns:", ", ".join(column_names))

# =====================================================
# MISSING VALUES
# =====================================================

missing_values = {}

for column in column_names:
    safe_column = '"' + column.replace('"', '""') + '"'

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM rain_data
        WHERE {safe_column} IS NULL
        """
    ).fetchone()[0]

    missing_values[column] = count

# =====================================================
# DUPLICATE RECORD CHECK
# =====================================================

duplicate_count = row_count - con.execute(
    """
    SELECT COUNT(*)
    FROM (
        SELECT DISTINCT *
        FROM rain_data
    ) AS unique_rows
    """
).fetchone()[0]

# =====================================================
# NUMERIC DESCRIPTIVE STATISTICS
# =====================================================

numeric_columns = [
    "Diameter_mm",
    "Velocity_m_s",
    "Volume_mm3"
]

stats = {}

for column in numeric_columns:
    result = con.execute(
        f"""
        SELECT
            COUNT("{column}"),
            AVG("{column}"),
            MEDIAN("{column}"),
            STDDEV_SAMP("{column}"),
            MIN("{column}"),
            MAX("{column}"),
            QUANTILE_CONT("{column}", 0.25),
            QUANTILE_CONT("{column}", 0.75)
        FROM rain_data
        """
    ).fetchone()

    stats[column] = {
        "count": result[0],
        "mean": result[1],
        "median": result[2],
        "std": result[3],
        "min": result[4],
        "max": result[5],
        "q1": result[6],
        "q3": result[7],
    }

# =====================================================
# INVALID MEASUREMENT CHECK
# =====================================================

invalid_measurements = con.execute(
    """
    SELECT COUNT(*)
    FROM rain_data
    WHERE
        Diameter_mm <= 0
        OR Velocity_m_s <= 0
        OR Volume_mm3 <= 0
    """
).fetchone()[0]

# =====================================================
# VOLUME FORMULA VALIDATION
# V = (PI / 6) * d^3
# =====================================================

formula_result = con.execute(
    """
    SELECT
        MAX(
            ABS(
                Volume_mm3
                - ((PI() / 6.0) * POWER(Diameter_mm, 3))
            )
        ),
        COUNT(
            CASE
                WHEN ABS(
                    Volume_mm3
                    - ((PI() / 6.0) * POWER(Diameter_mm, 3))
                ) > 0.0001
                THEN 1
            END
        )
    FROM rain_data
    """
).fetchone()

max_formula_error = formula_result[0]
formula_mismatches = formula_result[1]

# =====================================================
# CORRELATION
# =====================================================

correlation = con.execute(
    """
    SELECT CORR(Diameter_mm, Velocity_m_s)
    FROM rain_data
    """
).fetchone()[0]

# =====================================================
# OUTLIER ANALYSIS USING IQR
# =====================================================

outlier_results = {}

for column in numeric_columns:
    column_stats = stats[column]

    q1 = column_stats["q1"]
    q3 = column_stats["q3"]

    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    safe_column = '"' + column + '"'

    outlier_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM rain_data
        WHERE
            {safe_column} < ?
            OR {safe_column} > ?
        """,
        [lower_bound, upper_bound]
    ).fetchone()[0]

    outlier_results[column] = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower": lower_bound,
        "upper": upper_bound,
        "count": outlier_count,
        "percentage": (
            outlier_count / row_count * 100
            if row_count else 0
        )
    }

# =====================================================
# DATE RANGE AND DAILY RECORD COUNTS
# =====================================================

date_range = con.execute(
    """
    SELECT
        MIN(CAST(DateTime AS DATE)),
        MAX(CAST(DateTime AS DATE)),
        COUNT(DISTINCT CAST(DateTime AS DATE))
    FROM rain_data
    """
).fetchone()

daily_counts = con.execute(
    """
    SELECT
        CAST(DateTime AS DATE) AS observation_date,
        COUNT(*) AS records
    FROM rain_data
    GROUP BY observation_date
    ORDER BY observation_date
    """
).fetchall()

# =====================================================
# CONDITION-WISE RECORD COUNTS
# =====================================================

condition_counts = con.execute(
    """
    SELECT Condition, COUNT(*) AS records
    FROM rain_data
    GROUP BY Condition
    ORDER BY records DESC
    """
).fetchall()

# =====================================================
# GENERATE MARKDOWN REPORT
# =====================================================

lines = []

def add(text=""):
    lines.append(str(text))

def fmt(value, digits=6):
    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:,.{digits}f}"

    return f"{value:,}" if isinstance(value, int) else str(value)

add("# Rain Drop Volume Prediction Using AI")
add("## Exploratory Data Analysis (EDA) Report")
add()
add(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
add()
add("## 1. Objective")
add(
    "This report examines the deduplicated raindrop dataset, "
    "summarizes its measurements, checks data quality, identifies "
    "potential outliers, and assesses whether the data supports "
    "the proposed machine-learning objective."
)
add()
add("## 2. Dataset Overview")
add()
add(f"- **Dataset file:** `{DATASET.name}`")
add(f"- **Total records:** {row_count:,}")
add(f"- **Total columns:** {len(column_names)}")
add(f"- **First observation date:** {date_range[0]}")
add(f"- **Last observation date:** {date_range[1]}")
add(f"- **Distinct observation dates:** {date_range[2]}")
add()
add("### Column descriptions")
add()
add("| Column | Detected data type |")
add("|---|---|")

for column in columns:
    add(f"| {column[0]} | {column[1]} |")

add()
add("### Column interpretation")
add()
add("| Column | Interpretation |")
add("|---|---|")
add("| DateTime | Timestamp associated with an observation |")
add("| Diameter_mm | Raindrop diameter in millimetres |")
add("| Velocity_m_s | Velocity in metres per second |")
add("| Volume_mm3 | Volume in cubic millimetres |")
add("| Condition | Dataset condition/category label |")
add("| SourceFile | Source filename/category identifier |")
add()
add(
    "The meaning of each Condition label should be confirmed "
    "from the dataset documentation before drawing physical "
    "or experimental conclusions."
)
add()
add("## 3. Data Quality")
add()
add("| Check | Result |")
add("|---|---:|")
add(f"| Total records | {row_count:,} |")
add(f"| Exact duplicate rows | {duplicate_count:,} |")
add(f"| Invalid non-positive measurements | {invalid_measurements:,} |")

for column, count in missing_values.items():
    add(f"| Missing values — {column} | {count:,} |")

add()
add("### Data quality interpretation")
add()
if duplicate_count == 0:
    add("- No exact duplicate rows were detected in the deduplicated dataset.")
else:
    add(
        f"- {duplicate_count:,} exact duplicate rows remain; "
        "investigate before model training."
    )

total_missing = sum(missing_values.values())

if total_missing == 0:
    add("- No NULL values were detected in the listed columns.")
else:
    add(f"- NULL values detected across columns: {total_missing:,}.")

if invalid_measurements == 0:
    add("- No zero or negative diameter, velocity, or volume values were detected.")
else:
    add(
        f"- {invalid_measurements:,} rows contain at least one "
        "non-positive measurement."
    )

add()
add("## 4. Descriptive Statistics")
add()
add(
    "| Metric | Mean | Median | Std. deviation | "
    "Minimum | Maximum | Q1 | Q3 |"
)
add("|---|---:|---:|---:|---:|---:|---:|---:|")

for column in numeric_columns:
    s = stats[column]
    add(
        f"| {column} | {fmt(s['mean'])} | {fmt(s['median'])} | "
        f"{fmt(s['std'])} | {fmt(s['min'])} | {fmt(s['max'])} | "
        f"{fmt(s['q1'])} | {fmt(s['q3'])} |"
    )

add()
add("## 5. Outlier Analysis")
add()
add(
    "Potential outliers are identified using the Interquartile "
    "Range (IQR) rule: values below Q1 − 1.5 × IQR or above "
    "Q3 + 1.5 × IQR. An outlier is not automatically an error."
)
add()
add("| Metric | Lower bound | Upper bound | Potential outliers | Percentage |")
add("|---|---:|---:|---:|---:|")

for column, result in outlier_results.items():
    add(
        f"| {column} | {fmt(result['lower'])} | "
        f"{fmt(result['upper'])} | {result['count']:,} | "
        f"{result['percentage']:.2f}% |"
    )

add()
add("## 6. Correlation Analysis")
add()
add(
    f"- **Pearson correlation (Diameter vs Velocity):** "
    f"{fmt(correlation, 4)}"
)
add()
add(
    "Correlation describes linear association; it does not prove "
    "causation. The overall correlation may also hide differences "
    "between Condition groups."
)
add()
add("## 7. Volume Formula Validation")
add()
add("For an ideal spherical raindrop:")
add()
add("`Volume_mm3 = (PI / 6) × Diameter_mm³`")
add()
add(f"- Maximum absolute formula error: {fmt(max_formula_error, 12)} mm³")
add(f"- Rows with absolute error greater than 0.0001 mm³: {formula_mismatches:,}")
add()
if formula_mismatches == 0:
    add(
        "The volume values are consistent with the spherical-volume "
        "formula within the specified numerical tolerance."
    )
else:
    add(
        "Some rows differ from the formula beyond the selected "
        "tolerance and should be investigated."
    )

add()
add("## 8. Observation Dates")
add()
add("| Date | Record count |")
add("|---|---:|")

for day, count in daily_counts:
    add(f"| {day} | {count:,} |")

add()
add(
    "The date range alone does not establish continuous coverage. "
    "Check missing days, missing hours, measurement intervals, and "
    "whether records represent independent observations."
)
add()
add("## 9. Records by Condition")
add()
add("| Condition | Record count |")
add("|---|---:|")

for condition, count in condition_counts:
    add(f"| {condition} | {count:,} |")

add()
add("## 10. Key Findings")
add()
add(f"1. The deduplicated dataset contains **{row_count:,} records**.")
add(
    f"2. The dataset contains {len(column_names)} columns, including "
    "timestamp, diameter, velocity, volume, and two categorical identifiers."
)
add(
    f"3. The overall Pearson correlation between diameter and velocity "
    f"is {fmt(correlation, 4)}."
)
add(
    f"4. The maximum absolute error when validating the volume formula "
    f"is {fmt(max_formula_error, 12)} mm³."
)
add(
    "5. IQR-based outliers require domain review; unusually large "
    "measurements should not be removed automatically."
)
add()
add("## 11. Important Machine-Learning Consideration")
add()
add(
    "The volume column is mathematically derived from diameter using "
    "the spherical-volume formula. If the intended target is Volume_mm3 "
    "and the only input is Diameter_mm, a machine-learning model would "
    "largely reproduce a known equation rather than discover an "
    "independent predictive relationship."
)
add()
add(
    "A stronger forecasting objective would require a clearly defined "
    "future target, an appropriate time horizon, and relevant predictor "
    "variables. For rainfall forecasting, this might include historical "
    "rainfall or weather measurements, provided these exist and are "
    "documented. The current dataset spans a limited set of dates, so "
    "its suitability for general future forecasting must be assessed "
    "before model development."
)
add()
add("## 12. Recommended Next Steps")
add()
add("1. Verify the meaning of the Condition and SourceFile fields.")
add("2. Inspect the generated histograms and boxplots in this folder.")
add("3. Review outliers by Condition before deciding whether to retain them.")
add("4. Check temporal coverage and the sampling process.")
add("5. Define the exact prediction target and prediction horizon.")
add("6. Select a baseline model and use a leakage-safe evaluation strategy.")
add("7. Document all preprocessing decisions for reproducibility.")
add()
add("---")
add("*Generated automatically by `generate_eda_report.py`.*")

# =====================================================
# SAVE REPORT
# =====================================================

REPORT_FILE.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)

con.close()

print("\n" + "=" * 60)
print("EDA REPORT GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"\nRecords analyzed: {row_count:,}")
print(f"Report saved at:\n{REPORT_FILE}")
print("\nNext: Open reports/eda/EDA_Report.md in VS Code.")

