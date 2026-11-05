import duckdb

query = """
SELECT
    COUNT(*) AS total_rows,
    ROUND(
        CORR(Diameter_mm, Velocity_m_s),
        4
    ) AS diameter_velocity_correlation,
    ROUND(AVG(Diameter_mm), 4) AS avg_diameter_mm,
    ROUND(AVG(Velocity_m_s), 4) AS avg_velocity_m_s
FROM read_csv_auto(
    'data/raindrop_volume_deduplicated.csv'
)
"""

result = duckdb.sql(query).df()
print(result.to_string(index=False))