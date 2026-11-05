import duckdb

query = """
SELECT
    ROUND(CORR(Diameter_mm, Velocity_m_s), 4)
        AS diameter_velocity_corr,

    ROUND(CORR(Diameter_mm, Volume_mm3), 4)
        AS diameter_volume_corr,

    ROUND(CORR(Velocity_m_s, Volume_mm3), 4)
        AS velocity_volume_corr

FROM read_csv_auto(
    'data/raindrop_volume_deduplicated.csv'
)
"""

result = duckdb.sql(query).df()

print("\nOverall Feature Correlations:")
print(result.to_string(index=False))