import duckdb

query = """
SELECT
    MIN(DateTime) AS earliest_timestamp,
    MAX(DateTime) AS latest_timestamp,
    COUNT(DISTINCT CAST(DateTime AS DATE)) AS distinct_dates,
    COUNT(*) AS total_records
FROM read_csv_auto(
    'data/raindrop_volume_deduplicated.csv'
)
"""

result = duckdb.sql(query).df()

print("\nDataset DateTime Coverage:")
print(result.to_string(index=False))