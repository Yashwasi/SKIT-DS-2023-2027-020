import duckdb

query = """
SELECT
    CAST(DateTime AS DATE) AS date,
    COUNT(*) AS total_drops
FROM read_csv_auto(
    'data/raindrop_volume_deduplicated.csv'
)
GROUP BY CAST(DateTime AS DATE)
ORDER BY date
"""

result = duckdb.sql(query).df()

print(result.to_string(index=False))