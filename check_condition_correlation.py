import duckdb

query = """
SELECT
    Condition,
    COUNT(*) AS total_rows,
    ROUND(CORR(Diameter_mm, Velocity_m_s), 4) AS correlation
FROM read_csv_auto(
    'data/raindrop_volume_deduplicated.csv'
)
GROUP BY Condition
ORDER BY correlation DESC
"""

result = duckdb.sql(query).df()
print(result.to_string(index=False))