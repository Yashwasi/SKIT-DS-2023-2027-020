from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
)


# ==========================================
# CONFIGURATION
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

SAMPLE_SEEDS = [42, 100, 2026]


# ==========================================
# LOAD VALID OBSERVATIONS
# ==========================================

def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    print("Loading the deduplicated dataset...")

    df = pd.read_csv(
        DATA_FILE,
        usecols=FEATURES,
    )

    for column in FEATURES:
        df[column] = pd.to_numeric(
            df[column],
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
    ]

    df = df.reset_index(drop=True)

    print(f"Valid observations: {len(df):,}")

    return df


# ==========================================
# SAMPLE AND CLUSTER
# ==========================================

def cluster_sample(df, seed):

    sample = df.sample(
        n=SAMPLE_SIZE,
        random_state=seed,
        replace=False,
    ).reset_index(drop=True)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        sample[FEATURES]
    )

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
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

    result = {
        "Sample_Seed": seed,
        "Observations": len(sample),
        "Silhouette_Score": silhouette,
        "Davies_Bouldin_Index": db_index,
        "Inertia": model.inertia_,
        "Smallest_Cluster": int(counts.min()),
        "Largest_Cluster": int(counts.max()),
    }

    return result, sample, labels


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 60)
    print("INDEPENDENT SAMPLE VALIDATION")
    print("=" * 60)

    df = load_data()

    if len(df) < SAMPLE_SIZE:
        raise ValueError(
            f"Need at least {SAMPLE_SIZE} valid observations."
        )

    results = []

    for seed in SAMPLE_SEEDS:

        result, _, _ = cluster_sample(
            df,
            seed,
        )

        results.append(result)

        print(
            f"Sample seed {seed}: "
            f"Silhouette={result['Silhouette_Score']:.4f}, "
            f"DB Index={result['Davies_Bouldin_Index']:.4f}, "
            f"Smallest cluster={result['Smallest_Cluster']}, "
            f"Largest cluster={result['Largest_Cluster']}"
        )

    report = pd.DataFrame(results)

    output_file = (
        OUTPUT_DIR
        / "independent_sample_validation.csv"
    )

    report.to_csv(
        output_file,
        index=False,
    )

    print("\nVALIDATION SUMMARY")
    print("-" * 60)
    print(report.round(4).to_string(index=False))

    print("\nResults saved to:")
    print(output_file)

    print(
        "\nNote: This compares independently sampled subsets. "
        "Because each sample is standardized separately and "
        "cluster labels are arbitrary, these metrics are not "
        "a direct measure of label agreement across samples."
    )


if __name__ == "__main__":
    main()