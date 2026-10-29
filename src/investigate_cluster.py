
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "models" / "clustered_raindrops_sample.csv"

df = pd.read_csv(DATA_PATH)
cluster_2 = df[df["Cluster"] == 2].copy()

print("\n=== CLUSTER 2 SOURCE FILE COUNTS ===")
print(cluster_2["SourceFile"].value_counts().head(15))

print("\n=== CLUSTER 2 CONDITION COUNTS ===")
print(cluster_2["Condition"].value_counts().head(15))

print("\n=== SOURCE FILE DISTRIBUTION ACROSS CLUSTERS ===")
print(
    pd.crosstab(
        df["SourceFile"],
        df["Cluster"],
        normalize="index"
    ).round(3).head(15)
)

print("\n=== OBSERVATIONS WITH DIAMETER ABOVE 5 MM ===")
large = cluster_2[cluster_2["Diameter_mm"] > 5]

print("Count:", len(large))
print(
    large[
        [
            "DateTime", "Diameter_mm", "Velocity_m_s",
            "Volume_mm3", "Condition", "SourceFile"
        ]
    ].head(10).to_string(index=False)
)
