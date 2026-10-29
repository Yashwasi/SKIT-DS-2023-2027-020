
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "models" / "clustered_raindrops_sample.csv"
OUTPUT_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# DBSCAN settings to compare
EPS_VALUES = [0.3, 0.4, 0.5]
MIN_SAMPLES = 10

# --------------------------------------------------
# 2. Load the existing sample
# --------------------------------------------------
print("Loading clustered dataset sample...")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"File not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df = df.sample(
    n=min(15000, len(df)),
    random_state=42
).reset_index(drop=True)

features = ["Diameter_mm", "Velocity_m_s"]
df = df.dropna(subset=features + ["Volume_mm3"]).copy()

X = df[features].to_numpy()

print("Rows loaded:", len(df))

# --------------------------------------------------
# 3. Scale features
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 4. Compare DBSCAN parameter settings
# --------------------------------------------------
results = []
best_model = None
best_labels = None
best_score = -1
best_eps = None

rng = np.random.default_rng(RANDOM_STATE)
score_size = min(5000, len(df))
score_indices = rng.choice(
    len(df), size=score_size, replace=False
)

for eps in EPS_VALUES:
    print(f"\nTesting eps={eps}, min_samples={MIN_SAMPLES}")

    model = DBSCAN(
        eps=eps,
        min_samples=MIN_SAMPLES,
        n_jobs=-1
    )

    labels = model.fit_predict(X_scaled)

    non_noise = labels != -1
    cluster_labels = set(labels[non_noise])
    n_clusters = len(cluster_labels)
    noise_count = int((labels == -1).sum())
    noise_percentage = noise_count / len(labels) * 100

    # Silhouette is meaningful only when at least two
    # clusters exist and the evaluated sample contains
    # enough non-noise observations from those clusters.
    sampled_labels = labels[score_indices]
    sampled_X = X_scaled[score_indices]
    keep = sampled_labels != -1

    sampled_cluster_count = len(set(sampled_labels[keep]))
    silhouette = np.nan

    if (
        sampled_cluster_count >= 2
        and keep.sum() > sampled_cluster_count
        and keep.sum() > 1
    ):
        silhouette = silhouette_score(
            sampled_X[keep],
            sampled_labels[keep]
        )

    results.append({
        "eps": eps,
        "min_samples": MIN_SAMPLES,
        "Clusters_excluding_noise": n_clusters,
        "Noise_count": noise_count,
        "Noise_percentage": round(noise_percentage, 3),
        "Silhouette_excluding_noise": silhouette
    })

    print("Clusters excluding noise:", n_clusters)
    print("Noise observations:", noise_count)
    print("Noise percentage:", round(noise_percentage, 2), "%")
    print("Sample silhouette score:", silhouette)

    # Choose the best setting with a valid silhouette score.
    # The score is calculated on the same sampled rows.
    if np.isfinite(silhouette) and silhouette > best_score:
        best_score = silhouette
        best_model = model
        best_labels = labels.copy()
        best_eps = eps

# --------------------------------------------------
# 5. Save comparison results
# --------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUT_DIR / "dbscan_evaluation.csv",
    index=False
)

print("\nDBSCAN PARAMETER COMPARISON")
print(results_df.round(4).to_string(index=False))

if best_model is None:
    print(
        "\nNo tested setting produced a valid silhouette score. "
        "Review the parameter results before selecting a model."
    )
    raise SystemExit(0)

# --------------------------------------------------
# 6. Save labels and cluster summary
# --------------------------------------------------
df["DBSCAN_Cluster"] = best_labels

summary = df.groupby("DBSCAN_Cluster").agg(
    Number_of_Raindrops=("Volume_mm3", "size"),
    Mean_Diameter_mm=("Diameter_mm", "mean"),
    Mean_Velocity_m_s=("Velocity_m_s", "mean"),
    Mean_Volume_mm3=("Volume_mm3", "mean")
).round(4)

summary.to_csv(OUTPUT_DIR / "dbscan_cluster_summary.csv")

df.to_csv(
    OUTPUT_DIR / "dbscan_clustered_sample.csv",
    index=False
)

# Save model and scaler together
joblib.dump(
    {"scaler": scaler, "dbscan": best_model},
    OUTPUT_DIR / "dbscan_model.joblib"
)

# --------------------------------------------------
# 7. Plot DBSCAN groups and noise
# --------------------------------------------------
plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    df["Diameter_mm"],
    df["Velocity_m_s"],
    c=df["DBSCAN_Cluster"],
    s=7,
    alpha=0.5
)

plt.xlabel("Diameter (mm)")
plt.ylabel("Velocity (m/s)")
plt.title(f"DBSCAN Clusters (eps={best_eps})")
plt.colorbar(scatter, label="Cluster (-1 = noise)")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "dbscan_clusters.png",
    dpi=150
)
plt.close()

print("\nBEST TESTED EPS:", best_eps)
print("Best sampled silhouette score:", round(best_score, 4))
print("\nDBSCAN CLUSTER SUMMARY")
print(summary.to_string())
print("\nSaved:")
print("- models/dbscan_evaluation.csv")
print("- models/dbscan_cluster_summary.csv")
print("- models/dbscan_clustered_sample.csv")
print("- models/dbscan_clusters.png")
print("- models/dbscan_model.joblib")
