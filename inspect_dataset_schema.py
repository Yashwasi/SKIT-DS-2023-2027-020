import duckdb

query = """
DESCRIBE
SELECT *
FROM read_csv_auto(
    'data/raindrop_volume_deduplicated.csv'
)
"""

result = duckdb.sql(query).df()

print("\nDataset Schema:")
print(result.to_string(index=False))