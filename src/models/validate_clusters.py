from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "raindrop_volume_processed"
    / "raindrop_volume_deduplicated.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "cluster_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["Diameter_mm", "Velocity_m_s"]

SAMPLE_SIZE = 20_000
N_CLUSTERS = 3

RANDOM_STATE = 42

SEEDS = [0, 21, 42, 100, 2026]


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        usecols=FEATURES,
        nrows=SAMPLE_SIZE,
    )

    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=FEATURES,
    )

    df = df[
        (df["Diameter_mm"] > 0)
        & (df["Velocity_m_s"] > 0)
    ].copy()

    if len(df) < N_CLUSTERS:
        raise ValueError(
            "Insufficient valid rows for clustering."
        )

    return df.reset_index(drop=True)


# ==========================================
# 3. FIT AND EVALUATE K-MEANS
# ==========================================

def evaluate_model(X_scaled, seed):

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=seed,
        n_init=10,
    )

    labels = model.fit_predict(X_scaled)

    silhouette = silhouette_score(
        X_scaled,
        labels,
    )

    db_index = davies_bouldin_score(
        X_scaled,
        labels,
    )

    counts = pd.Series(labels).value_counts()

    return {
        "Random_State": seed,
        "Clusters": len(np.unique(labels)),
        "Silhouette_Score": silhouette,
        "Davies_Bouldin_Index": db_index,
        "Inertia": model.inertia_,
        "Smallest_Cluster": int(counts.min()),
        "Largest_Cluster": int(counts.max()),
        "Labels": labels,
    }


# ==========================================
# 4. MAIN VALIDATION
# ==========================================

def main():

    print("=" * 60)
    print("K-MEANS CLUSTER STABILITY VALIDATION")
    print("=" * 60)

    df = load_data()

    X = df[FEATURES].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Valid observations: {len(df):,}")
    print(f"Features: {FEATURES}")
    print(f"Clusters: {N_CLUSTERS}")

    results = []

    for seed in SEEDS:

        result = evaluate_model(
            X_scaled,
            seed,
        )

        print(
            f"Seed {seed}: "
            f"Silhouette={result['Silhouette_Score']:.4f}, "
            f"DB Index={result['Davies_Bouldin_Index']:.4f}, "
            f"Inertia={result['Inertia']:.4f}"
        )

        results.append(result)

    # Compare all runs with the first run using ARI.
    reference_labels = results[0]["Labels"]

    stability_results = []

    for result in results:

        ari = adjusted_rand_score(
            reference_labels,
            result["Labels"],
        )

        stability_results.append({
            "Random_State": result["Random_State"],
            "Silhouette_Score": result["Silhouette_Score"],
            "Davies_Bouldin_Index": result["Davies_Bouldin_Index"],
            "Inertia": result["Inertia"],
            "Smallest_Cluster": result["Smallest_Cluster"],
            "Largest_Cluster": result["Largest_Cluster"],
            "ARI_vs_Seed_0": ari,
        })

    report = pd.DataFrame(stability_results)

    output_file = (
        OUTPUT_DIR
        / "cluster_stability_results.csv"
    )

    report.to_csv(
        output_file,
        index=False,
    )

    print("\nSTABILITY RESULTS")
    print("-" * 80)
    print(report.round(4).to_string(index=False))

    print("\nSaved report:")
    print(output_file)

    print(
        "\nNote: This test changes K-Means initialization "
        "on the same sample. It does not measure stability "
        "across independent datasets."
    )


if __name__ == "__main__":
    main()