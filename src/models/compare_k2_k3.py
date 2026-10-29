
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
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "raindrop_volume_processed"
    / "raindrop_volume_deduplicated.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "cluster_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 20_000
RANDOM_STATE = 42

FEATURES = ["Diameter_mm", "Velocity_m_s"]

METRICS_FILE = OUTPUT_DIR / "k2_k3_metrics.csv"
PROFILES_FILE = OUTPUT_DIR / "k2_k3_cluster_profiles.csv"
ASSIGNMENTS_FILE = OUTPUT_DIR / "k2_k3_assignments.csv"


# ==========================================
# 2. LOAD AND VALIDATE DATA
# ==========================================

def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    print("Loading the first 20,000 rows...")

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

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES)

    df = df[
        (df["Diameter_mm"] > 0)
        & (df["Velocity_m_s"] > 0)
    ].copy()

    df = df.reset_index(drop=True)

    if len(df) < 3:
        raise ValueError("Not enough valid rows for comparison.")

    print(f"Valid observations: {len(df):,}")

    return df


# ==========================================
# 3. FIT K-MEANS MODELS
# ==========================================

def fit_models(df):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    results = {}
    metric_rows = []
    profile_rows = []

    for k in [2, 3]:

        print(f"\nTraining K-Means with K={k}...")

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        labels = model.fit_predict(X_scaled)

        counts = pd.Series(labels).value_counts().sort_index()

        silhouette = silhouette_score(
            X_scaled,
            labels,
        )

        db_index = davies_bouldin_score(
            X_scaled,
            labels,
        )

        metric_rows.append({
            "K": k,
            "Observations": len(df),
            "Silhouette_Score": silhouette,
            "Davies_Bouldin_Index": db_index,
            "Inertia": model.inertia_,
            "Smallest_Cluster": int(counts.min()),
            "Largest_Cluster": int(counts.max()),
            "Smallest_Cluster_Percent": (
                counts.min() / len(df) * 100
            ),
        })

        result = df.copy()
        result["Cluster"] = labels

        result["K"] = k

        results[k] = result

        for cluster_id, group in result.groupby("Cluster"):

            profile_rows.append({
                "K": k,
                "Cluster": int(cluster_id),
                "Count": len(group),
                "Percentage": len(group) / len(df) * 100,
                "Diameter_Mean": group["Diameter_mm"].mean(),
                "Diameter_Median": group["Diameter_mm"].median(),
                "Diameter_Min": group["Diameter_mm"].min(),
                "Diameter_Max": group["Diameter_mm"].max(),
                "Velocity_Mean": group["Velocity_m_s"].mean(),
                "Velocity_Median": group["Velocity_m_s"].median(),
                "Velocity_Min": group["Velocity_m_s"].min(),
                "Velocity_Max": group["Velocity_m_s"].max(),
            })

    metrics = pd.DataFrame(metric_rows)
    profiles = pd.DataFrame(profile_rows)

    metrics.to_csv(METRICS_FILE, index=False)
    profiles.round(6).to_csv(PROFILES_FILE, index=False)

    # Save side-by-side assignments for each observation.
    assignments = df.copy()
    assignments["Cluster_K2"] = results[2]["Cluster"].values
    assignments["Cluster_K3"] = results[3]["Cluster"].values

    assignments.to_csv(ASSIGNMENTS_FILE, index=False)

    return metrics, profiles, assignments


# ==========================================
# 4. COMPARE CLUSTER ASSIGNMENTS
# ==========================================

def compare_assignments(assignments):

    ari = adjusted_rand_score(
        assignments["Cluster_K2"],
        assignments["Cluster_K3"],
    )

    print("\nK=2 vs K=3 assignment comparison")
    print("-" * 60)
    print(
        "Adjusted Rand Index (ARI): "
        f"{ari:.4f}"
    )

    print(
        "Note: ARI compares the two partitions, "
        "but a low or moderate value is not automatically bad. "
        "K=2 and K=3 have different numbers of clusters."
    )


# ==========================================
# 5. PRINT RESULTS
# ==========================================

def print_results(metrics, profiles):

    print("\nMODEL METRICS")
    print("=" * 90)

    print(
        metrics.round(6).to_string(index=False)
    )

    print("\nCLUSTER PROFILES")
    print("=" * 90)

    print(
        profiles.round(4).to_string(index=False)
    )

    print("\nINTERPRETATION")

    k2 = metrics.loc[metrics["K"] == 2].iloc[0]
    k3 = metrics.loc[metrics["K"] == 3].iloc[0]

    if k3["Silhouette_Score"] > k2["Silhouette_Score"]:
        print("K=3 has the higher Silhouette Score.")
    else:
        print("K=2 has the higher Silhouette Score.")

    if k3["Davies_Bouldin_Index"] < k2["Davies_Bouldin_Index"]:
        print("K=3 has the lower Davies-Bouldin Index.")
    else:
        print("K=2 has the lower Davies-Bouldin Index.")

    print(
        "\nCluster size balance must also be considered. "
        "Do not select a model using one metric alone."
    )

    print("\nReports saved:")
    print(METRICS_FILE)
    print(PROFILES_FILE)
    print(ASSIGNMENTS_FILE)


# ==========================================
# 6. MAIN
# ==========================================

def main():

    print("=" * 65)
    print("K-MEANS COMPARISON: K=2 VS K=3")
    print("=" * 65)

    df = load_data()

    metrics, profiles, assignments = fit_models(df)

    compare_assignments(assignments)

    print_results(metrics, profiles)

    print("\nComparison completed successfully.")


if __name__ == "__main__":
    main()

