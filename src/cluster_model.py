
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "data" / "raindrop_volume_processed.zip"
OUTPUT_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_ROWS = 100_000
RANDOM_STATE = 42
MIN_K = 2
MAX_K = 8

# --------------------------------------------------
# 2. Load dataset sample
# --------------------------------------------------
print("Loading raindrop dataset...")

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {ZIP_PATH}")

df = pd.read_csv(
    ZIP_PATH,
    compression="zip",
    nrows=SAMPLE_ROWS
)

required_columns = [
    "Diameter_mm",
    "Velocity_m_s",
    "Volume_mm3"
]

missing = set(required_columns) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=required_columns)

df = df[
    (df["Diameter_mm"] > 0)
    & (df["Velocity_m_s"] > 0)
    & (df["Volume_mm3"] > 0)
].copy()

df.reset_index(drop=True, inplace=True)

if len(df) < MAX_K + 1:
    raise ValueError("Not enough valid rows for clustering.")

print("Valid rows:", len(df))

# --------------------------------------------------
# 3. Select and scale clustering features
# --------------------------------------------------
# Volume is analysed after clustering because it is
# mathematically derived from diameter.
features = ["Diameter_mm", "Velocity_m_s"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use a fixed subset to calculate comparable
# silhouette scores without excessive computation.
rng = np.random.default_rng(RANDOM_STATE)
score_size = min(5000, len(df))
score_indices = rng.choice(
    len(df), size=score_size, replace=False
)
X_score = X_scaled[score_indices]

# --------------------------------------------------
# 4. Compare different numbers of clusters
# --------------------------------------------------
print("\nEvaluating K values from 2 to 8...")

evaluation_results = []
trained_models = {}

for k in range(MIN_K, MAX_K + 1):
    print(f"Testing K = {k}")

    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=RANDOM_STATE
    )

    labels = kmeans.fit_predict(X_scaled)
    score_labels = labels[score_indices]

    silhouette = silhouette_score(
        X_score, score_labels
    )

    evaluation_results.append({
        "K": k,
        "Inertia": kmeans.inertia_,
        "Silhouette_Score": silhouette
    })

    trained_models[k] = kmeans

results = pd.DataFrame(evaluation_results)
results.to_csv(
    OUTPUT_DIR / "kmeans_evaluation.csv",
    index=False
)

print("\nK-MEANS EVALUATION")
print(results.round(4).to_string(index=False))

# --------------------------------------------------
# 5. Plot the Elbow Method
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(
    results["K"],
    results["Inertia"],
    marker="o"
)
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.xticks(results["K"])
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "elbow_method.png",
    dpi=150
)
plt.close()

# --------------------------------------------------
# 6. Plot silhouette scores
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(
    results["K"],
    results["Silhouette_Score"],
    marker="o"
)
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different K Values")
plt.xticks(results["K"])
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "silhouette_scores.png",
    dpi=150
)
plt.close()

# --------------------------------------------------
# 7. Choose K using the highest silhouette score
# --------------------------------------------------
best_row = results.loc[
    results["Silhouette_Score"].idxmax()
]
best_k = int(best_row["K"])
best_kmeans = trained_models[best_k]

print(f"\nBest K by silhouette score: {best_k}")
print(
    "Best silhouette score:",
    round(float(best_row["Silhouette_Score"]), 4)
)

# --------------------------------------------------
# 8. Save final pipeline and analyse volume
# --------------------------------------------------
df["Cluster"] = best_kmeans.labels_

summary = df.groupby("Cluster").agg(
    Number_of_Raindrops=("Volume_mm3", "size"),
    Mean_Diameter_mm=("Diameter_mm", "mean"),
    Mean_Velocity_m_s=("Velocity_m_s", "mean"),
    Mean_Volume_mm3=("Volume_mm3", "mean"),
    Median_Volume_mm3=("Volume_mm3", "median")
).round(4)

summary.to_csv(OUTPUT_DIR / "cluster_summary.csv")

final_pipeline = Pipeline([
    ("scaler", scaler),
    ("kmeans", best_kmeans)
])

joblib.dump(
    final_pipeline,
    OUTPUT_DIR / "raindrop_kmeans_model.joblib"
)

df.to_csv(
    OUTPUT_DIR / "clustered_raindrops_sample.csv",
    index=False
)

# --------------------------------------------------
# 9. Visualise final clusters
# --------------------------------------------------
plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    df["Diameter_mm"],
    df["Velocity_m_s"],
    c=df["Cluster"],
    s=7,
    alpha=0.45
)

plt.xlabel("Diameter (mm)")
plt.ylabel("Velocity (m/s)")
plt.title(f"Raindrop Clusters (K = {best_k})")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "raindrop_clusters.png",
    dpi=150
)
plt.close()

print("\nCLUSTER SUMMARY")
print(summary.to_string())

print("\nCompleted successfully!")
print("Evaluation: models/kmeans_evaluation.csv")
print("Elbow graph: models/elbow_method.png")
print("Silhouette graph: models/silhouette_scores.png")
print("Cluster graph: models/raindrop_clusters.png")
print("Cluster summary: models/cluster_summary.csv")
print("Saved model: models/raindrop_kmeans_model.joblib")
