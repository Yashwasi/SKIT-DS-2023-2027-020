import duckdb

query = """
SELECT
    Condition,
    COUNT(DISTINCT SourceFile) AS source_file_count,
    COUNT(*) AS total_rows
FROM read_csv_auto(
    'data/raindrop_volume_deduplicated.csv'
)
GROUP BY Condition
ORDER BY Condition
"""

result = duckdb.sql(query).df()

print(result.to_string(index=False))