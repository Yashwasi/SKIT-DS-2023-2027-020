import duckdb

query = """
SELECT
    Condition,
    COUNT(*) AS total_rows,

    COUNT(*) FILTER (
        WHERE Diameter_mm > 2
    ) AS above_2mm,

    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE Diameter_mm > 2
        ) / COUNT(*),
        4
    ) AS pct_above_2mm,

    COUNT(*) FILTER (
        WHERE Diameter_mm > 3
    ) AS above_3mm,

    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE Diameter_mm > 3
        ) / COUNT(*),
        4
    ) AS pct_above_3mm,

    COUNT(*) FILTER (
        WHERE Diameter_mm > 5
    ) AS above_5mm,

    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE Diameter_mm > 5
        ) / COUNT(*),
        4
    ) AS pct_above_5mm

FROM read_csv_auto(
    'data/raindrop_volume_deduplicated.csv'
)

GROUP BY Condition
ORDER BY pct_above_5mm DESC
"""

result = duckdb.sql(query).df()

print(result.to_string(index=False))