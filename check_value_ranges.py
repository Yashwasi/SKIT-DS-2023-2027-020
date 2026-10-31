import duckdb

query = """
SELECT
    Condition,
    COUNT(*) AS total_rows,

    COUNT(*) FILTER (
        WHERE Diameter_mm > 2
    ) AS diameter_above_2mm,

    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE Diameter_mm > 2
        ) / COUNT(*),
        4
    ) AS percentage_above_2mm

FROM read_csv_auto(
    'data/raindrop_volume_deduplicated.csv'
)

GROUP BY Condition

ORDER BY percentage_above_2mm DESC
"""

result = duckdb.sql(query).df()

print(result.to_string(index=False))