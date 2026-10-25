
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "data" / "raindrop_volume_processed.zip"
OUTPUT_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_SIZE = 15_000
RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load and prepare the same sample for both models
# --------------------------------------------------
print("Loading dataset...")

df = pd.read_csv(
    ZIP_PATH,
    compression="zip",
    nrows=100_000
)

features = ["Diameter_mm", "Velocity_m_s", "Volume_mm3"]

for column in features:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features)

df = df[
    (df["Diameter_mm"] > 0)
    & (df["Velocity_m_s"] > 0)
    & (df["Volume_mm3"] > 0)
].copy()

# Reproducible sample shared by both algorithms
df = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("Rows used for both models:", len(df))

# --------------------------------------------------
# 3. Use diameter and velocity only
# --------------------------------------------------
# Volume is derived from diameter, so we exclude it
# from clustering to avoid giving size extra influence.
X = df[["Diameter_mm", "Velocity_m_s"]].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 4. Fit K-Means
# --------------------------------------------------
print("\nRunning K-Means...")

kmeans = KMeans(
    n_clusters=3,
    n_init=10,
    random_state=RANDOM_STATE
)

kmeans_labels = kmeans.fit_predict(X_scaled)

# --------------------------------------------------
# 5. Fit DBSCAN
# --------------------------------------------------
print("Running DBSCAN...")

dbscan = DBSCAN(
    eps=0.5,
    min_samples=10,
    n_jobs=-1
)

dbscan_labels = dbscan.fit_predict(X_scaled)

# --------------------------------------------------
# 6. Calculate comparable metrics
# --------------------------------------------------
def evaluate_clustering(X_values, labels):
    labels = np.asarray(labels)

    noise_mask = labels == -1
    valid_mask = ~noise_mask

    valid_labels = labels[valid_mask]
    unique_clusters = np.unique(valid_labels)
    n_clusters = len(unique_clusters)

    noise_count = int(noise_mask.sum())
    noise_percentage = noise_count / len(labels) * 100

    score = np.nan

    # Silhouette calculated on non-noise points for
    # both algorithms, using the same feature space.
    if (
        n_clusters >= 2
        and valid_mask.sum() > n_clusters
    ):
        score = silhouette_score(
            X_values[valid_mask],
            valid_labels
        )

    return {
        "Clusters_excluding_noise": n_clusters,
        "Noise_count": noise_count,
        "Noise_percentage": round(noise_percentage, 3),
        "Silhouette_excluding_noise": score
    }

results = pd.DataFrame({
    "K-Means": evaluate_clustering(X_scaled, kmeans_labels),
    "DBSCAN": evaluate_clustering(X_scaled, dbscan_labels)
}).T

print("\nMODEL COMPARISON")
print(results.round(4).to_string())

results.to_csv(OUTPUT_DIR / "model_comparison.csv")

# --------------------------------------------------
# 7. Save cluster labels and summaries
# --------------------------------------------------
df["KMeans_Cluster"] = kmeans_labels
df["DBSCAN_Cluster"] = dbscan_labels

df.to_csv(
    OUTPUT_DIR / "model_comparison_predictions.csv",
    index=False
)

for algorithm, label_column in [
    ("KMeans", "KMeans_Cluster"),
    ("DBSCAN", "DBSCAN_Cluster")
]:
    summary = df.groupby(label_column).agg(
        Number_of_Raindrops=("Diameter_mm", "size"),
        Mean_Diameter_mm=("Diameter_mm", "mean"),
        Mean_Velocity_m_s=("Velocity_m_s", "mean"),
        Mean_Volume_mm3=("Volume_mm3", "mean")
    ).round(4)

    summary.to_csv(
        OUTPUT_DIR / f"{algorithm.lower()}_comparison_summary.csv"
    )

# --------------------------------------------------
# 8. Final output
# --------------------------------------------------
print("\nComparison completed successfully.")
print("Saved: models/model_comparison.csv")
print("Saved: models/model_comparison_predictions.csv")
print("Saved: models/kmeans_comparison_summary.csv")
print("Saved: models/dbscan_comparison_summary.csv")
