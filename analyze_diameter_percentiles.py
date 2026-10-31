import duckdb

query = """
SELECT
    Condition,
    COUNT(*) AS total_rows,

    ROUND(
        QUANTILE_CONT(Diameter_mm, 0.50), 4
    ) AS percentile_50,

    ROUND(
        QUANTILE_CONT(Diameter_mm, 0.90), 4
    ) AS percentile_90,

    ROUND(
        QUANTILE_CONT(Diameter_mm, 0.95), 4
    ) AS percentile_95,

    ROUND(
        QUANTILE_CONT(Diameter_mm, 0.99), 4
    ) AS percentile_99

FROM read_csv_auto(
    'data/raindrop_volume_deduplicated.csv'
)

GROUP BY Condition
ORDER BY Condition
"""

result = duckdb.sql(query).df()

print(result.to_string(index=False))