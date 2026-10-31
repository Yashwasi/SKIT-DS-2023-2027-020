import duckdb

query = """
SELECT
    DATE_TRUNC('hour', DateTime) AS hour,
    COUNT(*) AS total_drops
FROM read_csv_auto(
    'data/raindrop_volume_deduplicated.csv'
)
GROUP BY DATE_TRUNC('hour', DateTime)
ORDER BY hour
"""

result = duckdb.sql(query).df()

print(result.to_string(index=False))