
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "raindrop_volume_processed"
    / "raindrop_volume_deduplicated.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 20_000
FEATURE_COLUMNS = ["Diameter_mm", "Velocity_m_s"]


def load_sample():
    """Load and validate the sample used for model evaluation."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        usecols=FEATURE_COLUMNS,
        nrows=SAMPLE_SIZE,
    )

    if df.empty:
        raise ValueError("Dataset sample is empty.")

    if df.isnull().any().any():
        raise ValueError(
            "Missing values found in selected features."
        )

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(
            "Non-finite values found in selected features."
        )

    if (df <= 0).any().any():
        raise ValueError(
            "Non-positive feature values found."
        )

    return df


def evaluate_kmeans(X):
    """Evaluate K-Means for k values from 2 through 8."""
    results = []

    for k in range(2, 9):
        print(f"Evaluating K-Means: k={k}")

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(X)
        counts = pd.Series(labels).value_counts()

        results.append({
            "Algorithm": "KMeans",
            "Parameters": f"k={k}",
            "Clusters": len(np.unique(labels)),
            "Noise_Percent": 0.0,
            "Silhouette_Score": silhouette_score(X, labels),
            "Davies_Bouldin_Index": davies_bouldin_score(
                X, labels
            ),
            "Inertia": model.inertia_,
            "Smallest_Cluster": int(counts.min()),
            "Largest_Cluster": int(counts.max()),
        })

    return pd.DataFrame(results)


def evaluate_dbscan(X):
    """Evaluate DBSCAN using selected eps values."""
    results = []

    eps_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    min_samples = 10

    for eps in eps_values:
        print(f"Evaluating DBSCAN: eps={eps}")

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=-1,
        )

        labels = model.fit_predict(X)

        noise_mask = labels == -1
        non_noise_mask = ~noise_mask

        non_noise_labels = labels[non_noise_mask]
        cluster_count = len(np.unique(non_noise_labels))

        silhouette = np.nan
        db_index = np.nan

        if (
            cluster_count >= 2
            and cluster_count < len(non_noise_labels)
        ):
            X_valid = X[non_noise_mask]

            silhouette = silhouette_score(
                X_valid,
                non_noise_labels,
            )

            db_index = davies_bouldin_score(
                X_valid,
                non_noise_labels,
            )

        results.append({
            "Algorithm": "DBSCAN",
            "Parameters": (
                f"eps={eps}, min_samples={min_samples}"
            ),
            "Clusters": cluster_count,
            "Noise_Percent": noise_mask.mean() * 100,
            "Silhouette_Score": silhouette,
            "Davies_Bouldin_Index": db_index,
            "Inertia": np.nan,
            "Smallest_Cluster": (
                int(pd.Series(non_noise_labels).value_counts().min())
                if cluster_count > 0
                else 0
            ),
            "Largest_Cluster": (
                int(pd.Series(non_noise_labels).value_counts().max())
                if cluster_count > 0
                else 0
            ),
        })

    return pd.DataFrame(results)


def main():
    print("Loading evaluation sample...")
    df = load_sample()

    print(f"Rows loaded: {len(df):,}")

    # Use original features, based on preliminary experiments.
    # Fit the scaler on this evaluation sample only.
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLUMNS])

    kmeans_results = evaluate_kmeans(X)
    dbscan_results = evaluate_dbscan(X)

    all_results = pd.concat(
        [kmeans_results, dbscan_results],
        ignore_index=True,
    )

    kmeans_path = OUTPUT_DIR / "kmeans_evaluation.csv"
    dbscan_path = OUTPUT_DIR / "dbscan_evaluation.csv"
    combined_path = OUTPUT_DIR / "combined_evaluation.csv"

    kmeans_results.to_csv(kmeans_path, index=False)
    dbscan_results.to_csv(dbscan_path, index=False)
    all_results.to_csv(combined_path, index=False)

    print("\nK-MEANS RESULTS")
    print("=" * 75)
    print(kmeans_results.round(4).to_string(index=False))

    print("\nDBSCAN RESULTS")
    print("=" * 75)
    print(dbscan_results.round(4).to_string(index=False))

    print("\nSaved evaluation reports:")
    print(kmeans_path)
    print(dbscan_path)
    print(combined_path)

    print(
        "\nEvaluation complete. Results are preliminary "
        "and based on a sample, not the full dataset."
    )


if __name__ == "__main__":
    main()

