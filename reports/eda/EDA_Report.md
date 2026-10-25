# Rain Drop Volume Prediction Using AI
## Exploratory Data Analysis (EDA) Report

**Generated:** 2026-10-25 15:27:28

## 1. Objective
This report examines the deduplicated raindrop dataset, summarizes its measurements, checks data quality, identifies potential outliers, and assesses whether the data supports the proposed machine-learning objective.

## 2. Dataset Overview

- **Dataset file:** `raindrop_volume_deduplicated.csv`
- **Total records:** 12,461,324
- **Total columns:** 6
- **First observation date:** 2016-06-13
- **Last observation date:** 2016-06-20
- **Distinct observation dates:** 6

### Column descriptions

| Column | Detected data type |
|---|---|
| DateTime | TIMESTAMP |
| Diameter_mm | DOUBLE |
| Velocity_m_s | DOUBLE |
| Volume_mm3 | DOUBLE |
| Condition | VARCHAR |
| SourceFile | VARCHAR |

### Column interpretation

| Column | Interpretation |
|---|---|
| DateTime | Timestamp associated with an observation |
| Diameter_mm | Raindrop diameter in millimetres |
| Velocity_m_s | Velocity in metres per second |
| Volume_mm3 | Volume in cubic millimetres |
| Condition | Dataset condition/category label |
| SourceFile | Source filename/category identifier |

The meaning of each Condition label should be confirmed from the dataset documentation before drawing physical or experimental conclusions.

## 3. Data Quality

| Check | Result |
|---|---:|
| Total records | 12,461,324 |
| Exact duplicate rows | 0 |
| Invalid non-positive measurements | 0 |
| Missing values — DateTime | 0 |
| Missing values — Diameter_mm | 0 |
| Missing values — Velocity_m_s | 0 |
| Missing values — Volume_mm3 | 0 |
| Missing values — Condition | 0 |
| Missing values — SourceFile | 0 |

### Data quality interpretation

- No exact duplicate rows were detected in the deduplicated dataset.
- No NULL values were detected in the listed columns.
- No zero or negative diameter, velocity, or volume values were detected.

## 4. Descriptive Statistics

| Metric | Mean | Median | Std. deviation | Minimum | Maximum | Q1 | Q3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Diameter_mm | 0.767882 | 0.700000 | 0.315067 | 0.520000 | 9.689000 | 0.624000 | 0.825000 |
| Velocity_m_s | 2.311720 | 2.021000 | 1.040023 | 1.000000 | 11.999000 | 1.528000 | 2.920000 |
| Volume_mm3 | 0.497723 | 0.179594 | 3.963808 | 0.073622 | 476.250550 | 0.127219 | 0.294009 |

## 5. Outlier Analysis

Potential outliers are identified using the Interquartile Range (IQR) rule: values below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR. An outlier is not automatically an error.

| Metric | Lower bound | Upper bound | Potential outliers | Percentage |
|---|---:|---:|---:|---:|
| Diameter_mm | 0.322500 | 1.126500 | 618,082 | 4.96% |
| Velocity_m_s | -0.560000 | 5.008000 | 211,222 | 1.70% |
| Volume_mm3 | -0.122966 | 0.544194 | 1,041,826 | 8.36% |

## 6. Correlation Analysis

- **Pearson correlation (Diameter vs Velocity):** 0.1933

Correlation describes linear association; it does not prove causation. The overall correlation may also hide differences between Condition groups.

## 7. Volume Formula Validation

For an ideal spherical raindrop:

`Volume_mm3 = (PI / 6) × Diameter_mm³`

- Maximum absolute formula error: 0.000000000000 mm³
- Rows with absolute error greater than 0.0001 mm³: 0

The volume values are consistent with the spherical-volume formula within the specified numerical tolerance.

## 8. Observation Dates

| Date | Record count |
|---|---:|
| 2016-06-13 | 1,669,161 |
| 2016-06-14 | 4,114,305 |
| 2016-06-15 | 283,877 |
| 2016-06-16 | 1,854,437 |
| 2016-06-17 | 4,168,036 |
| 2016-06-20 | 371,508 |

The date range alone does not establish continuous coverage. Check missing days, missing hours, measurement intervals, and whether records represent independent observations.

## 9. Records by Condition

| Condition | Record count |
|---|---:|
| Foliated_Cedar1c_040 | 1,751,358 |
| Foliated_Cedar1a_040 | 1,622,328 |
| Foliated_Cedar1c_100 | 1,531,525 |
| Foliated_Cedar1a_100 | 1,353,389 |
| Foliated_Cedar1b_100 | 1,283,122 |
| Foliated_Cedar1c_030 | 772,491 |
| Foliated_Cedar1b_040 | 740,619 |
| Foliated_Cedar1c_050 | 732,655 |
| Foliated_Cedar1a_030 | 592,839 |
| Foliated_Cedar1a_050 | 577,778 |
| Foliated_Cedar1b_050 | 544,004 |
| Foliated_Cedar1b_020 | 326,295 |
| Foliated_Cedar1b_030 | 303,831 |
| Foliated_Cedar1a_015 | 129,948 |
| Foliated_Cedar1c_015 | 108,814 |
| Foliated_Cedar1c_020 | 45,213 |
| Foliated_Cedar1b_015 | 45,115 |

## 10. Key Findings

1. The deduplicated dataset contains **12,461,324 records**.
2. The dataset contains 6 columns, including timestamp, diameter, velocity, volume, and two categorical identifiers.
3. The overall Pearson correlation between diameter and velocity is 0.1933.
4. The maximum absolute error when validating the volume formula is 0.000000000000 mm³.
5. IQR-based outliers require domain review; unusually large measurements should not be removed automatically.

## 11. Important Machine-Learning Consideration

The volume column is mathematically derived from diameter using the spherical-volume formula. If the intended target is Volume_mm3 and the only input is Diameter_mm, a machine-learning model would largely reproduce a known equation rather than discover an independent predictive relationship.

A stronger forecasting objective would require a clearly defined future target, an appropriate time horizon, and relevant predictor variables. For rainfall forecasting, this might include historical rainfall or weather measurements, provided these exist and are documented. The current dataset spans a limited set of dates, so its suitability for general future forecasting must be assessed before model development.

## 12. Recommended Next Steps

1. Verify the meaning of the Condition and SourceFile fields.
2. Inspect the generated histograms and boxplots in this folder.
3. Review outliers by Condition before deciding whether to retain them.
4. Check temporal coverage and the sampling process.
5. Define the exact prediction target and prediction horizon.
6. Select a baseline model and use a leakage-safe evaluation strategy.
7. Document all preprocessing decisions for reproducibility.

---
*Generated automatically by `generate_eda_report.py`.*
