
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
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
    / "independent_k2_k3_validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["Diameter_mm", "Velocity_m_s"]

SAMPLE_SIZE = 20_000
SAMPLE_SEEDS = [42, 100, 2026]
CLUSTER_COUNTS = [2, 3]

MODEL_SEED = 42
N_INIT = 10


# ==========================================
# 2. LOAD VALID OBSERVATIONS
# ==========================================

def load_valid_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    print("Loading valid observations in chunks...")

    chunks = []

    for chunk in pd.read_csv(
        DATA_FILE,
        usecols=FEATURES,
        chunksize=500_000,
    ):

        for column in FEATURES:
            chunk[column] = pd.to_numeric(
                chunk[column],
                errors="coerce",
            )

        chunk = chunk.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        chunk = chunk.dropna(subset=FEATURES)

        chunk = chunk[
            (chunk["Diameter_mm"] > 0)
            & (chunk["Velocity_m_s"] > 0)
        ]

        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        raise ValueError(
            "No valid observations found in the dataset."
        )

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    print(f"Valid observations available: {len(df):,}")

    if len(df) < SAMPLE_SIZE:
        raise ValueError(
            f"At least {SAMPLE_SIZE:,} valid observations required."
        )

    return df


# ==========================================
# 3. SAMPLE WITHOUT REPLACEMENT
# ==========================================

def create_sample(df, seed):

    sample = df.sample(
        n=SAMPLE_SIZE,
        replace=False,
        random_state=seed,
    )

    return sample.reset_index(drop=True)


# ==========================================
# 4. TRAIN AND EVALUATE MODELS
# ==========================================

def evaluate_sample(sample, sample_seed):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        sample[FEATURES]
    )

    metrics_rows = []
    profiles_rows = []

    for k in CLUSTER_COUNTS:

        print(
            f"  Sample seed={sample_seed}, K={k}: training..."
        )

        model = KMeans(
            n_clusters=k,
            random_state=MODEL_SEED,
            n_init=N_INIT,
        )

        labels = model.fit_predict(X_scaled)

        counts = pd.Series(labels).value_counts()

        metrics_rows.append({
            "Sample_Seed": sample_seed,
            "Sample_Size": len(sample),
            "K": k,
            "Silhouette_Score": silhouette_score(
                X_scaled,
                labels,
            ),
            "Davies_Bouldin_Index": davies_bouldin_score(
                X_scaled,
                labels,
            ),
            "Inertia": model.inertia_,
            "Smallest_Cluster": int(counts.min()),
            "Largest_Cluster": int(counts.max()),
            "Smallest_Cluster_Percent": (
                counts.min() / len(sample) * 100
            ),
        })

        result = sample.copy()
        result["Cluster"] = labels

        for cluster_id, group in result.groupby("Cluster"):

            profiles_rows.append({
                "Sample_Seed": sample_seed,
                "K": k,
                "Cluster": int(cluster_id),
                "Count": len(group),
                "Percentage": len(group) / len(sample) * 100,
                "Diameter_Mean": group["Diameter_mm"].mean(),
                "Diameter_Median": group["Diameter_mm"].median(),
                "Diameter_Min": group["Diameter_mm"].min(),
                "Diameter_Max": group["Diameter_mm"].max(),
                "Velocity_Mean": group["Velocity_m_s"].mean(),
                "Velocity_Median": group["Velocity_m_s"].median(),
                "Velocity_Min": group["Velocity_m_s"].min(),
                "Velocity_Max": group["Velocity_m_s"].max(),
            })

    return metrics_rows, profiles_rows


# ==========================================
# 5. MAIN
# ==========================================

def main():

    print("=" * 70)
    print("INDEPENDENT SAMPLE VALIDATION: K=2 VS K=3")
    print("=" * 70)

    df = load_valid_data()

    all_metrics = []
    all_profiles = []

    for seed in SAMPLE_SEEDS:

        print(f"\nCreating independent sample: seed={seed}")

        sample = create_sample(df, seed)

        metrics, profiles = evaluate_sample(
            sample,
            seed,
        )

        all_metrics.extend(metrics)
        all_profiles.extend(profiles)

    metrics_df = pd.DataFrame(all_metrics)
    profiles_df = pd.DataFrame(all_profiles)

    metrics_file = (
        OUTPUT_DIR / "independent_k2_k3_metrics.csv"
    )

    profiles_file = (
        OUTPUT_DIR / "independent_k2_k3_profiles.csv"
    )

    metrics_df.to_csv(
        metrics_file,
        index=False,
    )

    profiles_df.to_csv(
        profiles_file,
        index=False,
    )

    print("\n" + "=" * 70)
    print("VALIDATION METRICS")
    print("=" * 70)

    print(
        metrics_df.round(5).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("CLUSTER PROFILES")
    print("=" * 70)

    print(
        profiles_df.round(4).to_string(index=False)
    )

    print("\nREPORTS SAVED")
    print(metrics_file)
    print(profiles_file)

    print(
        "\nValidation completed. Compare metrics and cluster profiles "
        "across samples before selecting a final configuration."
    )


if __name__ == "__main__":
    main()

