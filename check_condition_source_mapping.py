import duckdb

query = """
SELECT
    Condition,
    SourceFile,
    COUNT(*) AS total_rows
FROM read_csv_auto(
    'data/raindrop_volume_deduplicated.csv'
)
GROUP BY Condition, SourceFile
ORDER BY Condition
"""

result = duckdb.sql(query).df()

print("\nCondition-to-SourceFile Mapping:")
print(result.to_string(index=False))
