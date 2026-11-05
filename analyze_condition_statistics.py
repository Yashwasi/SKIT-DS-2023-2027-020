import duckdb

query = """
SELECT
    Condition,
    COUNT(*) AS total_rows,

    ROUND(AVG(Diameter_mm), 4) AS avg_diameter_mm,
    ROUND(MEDIAN(Diameter_mm), 4) AS median_diameter_mm,

    ROUND(AVG(Velocity_m_s), 4) AS avg_velocity_m_s,
    ROUND(MEDIAN(Velocity_m_s), 4) AS median_velocity_m_s,

    ROUND(AVG(Volume_mm3), 4) AS avg_volume_mm3,

    ROUND(MIN(Diameter_mm), 4) AS min_diameter_mm,
    ROUND(MAX(Diameter_mm), 4) AS max_diameter_mm

FROM read_csv_auto(
    'data/raindrop_volume_deduplicated.csv'
)

GROUP BY Condition
ORDER BY total_rows DESC
"""

result = duckdb.sql(query).df()

print(result.to_string(index=False))