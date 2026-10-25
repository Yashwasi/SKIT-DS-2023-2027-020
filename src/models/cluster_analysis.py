
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
N_CLUSTERS = 3

FEATURES = ["Diameter_mm", "Velocity_m_s"]

CLUSTER_COUNTS_TO_COMPARE = [2, 3, 4]

# Maximum observations retained in the extreme-value report.
EXTREME_ROWS = 100


# ==========================================
# 2. LOAD AND VALIDATE DATA
# ==========================================

def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    print("Loading dataset sample...")

    df = pd.read_csv(
        DATA_FILE,
        usecols=FEATURES,
        nrows=SAMPLE_SIZE,
    )

    print(f"Rows loaded: {len(df):,}")

    for column in FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    missing_before = df[FEATURES].isna().sum()

    invalid_mask = (
        (df["Diameter_mm"] <= 0)
        | (df["Velocity_m_s"] <= 0)
    )

    invalid_count = int(invalid_mask.sum())

    df = df.dropna(subset=FEATURES)

    df = df[
        (df["Diameter_mm"] > 0)
        & (df["Velocity_m_s"] > 0)
    ].copy()

    if len(df) < max(CLUSTER_COUNTS_TO_COMPARE):
        raise ValueError(
            "Not enough valid observations for clustering."
        )

    print("\nDATA VALIDATION")
    print("-" * 60)
    print(f"Missing/non-numeric values by feature:\n{missing_before}")
    print(f"Rows with non-positive values: {invalid_count:,}")
    print(f"Valid rows retained: {len(df):,}")

    return df.reset_index(drop=True)


# ==========================================
# 3. SCALE FEATURES
# ==========================================

def scale_features(df):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(df[FEATURES])

    print("\nFeatures standardized successfully.")

    return X_scaled


# ==========================================
# 4. TRAIN PRIMARY K-MEANS MODEL
# ==========================================

def train_model(X_scaled):

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    labels = model.fit_predict(X_scaled)

    print("K-Means clustering completed.")
    print(f"Number of clusters: {N_CLUSTERS}")

    return model, labels


# ==========================================
# 5. CLUSTER PROFILES AND ASSIGNMENTS
# ==========================================

def analyze_clusters(df, labels):

    result = df.copy()
    result["Cluster"] = labels

    total_rows = len(result)

    profile = (
        result.groupby("Cluster")
        .agg(
            Count=("Diameter_mm", "size"),
            Diameter_Mean=("Diameter_mm", "mean"),
            Diameter_Median=("Diameter_mm", "median"),
            Diameter_Min=("Diameter_mm", "min"),
            Diameter_Max=("Diameter_mm", "max"),
            Velocity_Mean=("Velocity_m_s", "mean"),
            Velocity_Median=("Velocity_m_s", "median"),
            Velocity_Min=("Velocity_m_s", "min"),
            Velocity_Max=("Velocity_m_s", "max"),
        )
        .reset_index()
    )

    profile["Percentage"] = (
        profile["Count"] / total_rows * 100
    )

    profile_file = OUTPUT_DIR / "cluster_profiles.csv"
    profile.round(4).to_csv(profile_file, index=False)

    assignments_file = (
        OUTPUT_DIR / "cluster_assignments.csv"
    )
    result.to_csv(assignments_file, index=False)

    print("\nCLUSTER PROFILE")
    print("=" * 90)
    print(profile.round(4).to_string(index=False))

    print(f"\nCluster profiles saved: {profile_file}")
    print(f"Cluster assignments saved: {assignments_file}")

    return result, profile


# ==========================================
# 6. IQR OUTLIER INVESTIGATION
# ==========================================

def investigate_outliers(result):

    print("\nOUTLIER INVESTIGATION")
    print("-" * 60)

    outlier_flags = result.copy()

    feature_summary = []

    for column in FEATURES:

        q1 = result[column].quantile(0.25)
        q3 = result[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        flag_column = f"{column}_IQR_Outlier"

        outlier_flags[flag_column] = (
            (result[column] < lower_bound)
            | (result[column] > upper_bound)
        )

        count = int(outlier_flags[flag_column].sum())

        feature_summary.append({
            "Feature": column,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
            "Outlier_Count": count,
            "Outlier_Percentage": count / len(result) * 100,
        })

        print(
            f"{column}: {count:,} IQR outliers "
            f"({count / len(result) * 100:.2f}%)"
        )

    feature_summary_df = pd.DataFrame(feature_summary)

    feature_summary_file = (
        OUTPUT_DIR / "iqr_outlier_summary.csv"
    )

    feature_summary_df.round(6).to_csv(
        feature_summary_file,
        index=False,
    )

    # Count observations outside the IQR bounds of either feature.
    flag_columns = [
        f"{column}_IQR_Outlier"
        for column in FEATURES
    ]

    outlier_flags["Any_IQR_Outlier"] = (
        outlier_flags[flag_columns].any(axis=1)
    )

    # Analyze outliers by cluster.
    cluster_outlier_summary = (
        outlier_flags.groupby("Cluster")
        .agg(
            Total_Observations=("Cluster", "size"),
            Diameter_Outliers=(
                "Diameter_mm_IQR_Outlier", "sum"
            ),
            Velocity_Outliers=(
                "Velocity_m_s_IQR_Outlier", "sum"
            ),
            Any_Feature_Outlier=(
                "Any_IQR_Outlier", "sum"
            ),
        )
        .reset_index()
    )

    cluster_outlier_summary["Any_Outlier_Percentage"] = (
        cluster_outlier_summary["Any_Feature_Outlier"]
        / cluster_outlier_summary["Total_Observations"]
        * 100
    )

    cluster_outlier_file = (
        OUTPUT_DIR / "cluster_outlier_summary.csv"
    )

    cluster_outlier_summary.round(4).to_csv(
        cluster_outlier_file,
        index=False,
    )

    # Save observations with unusual values for manual inspection.
    outlier_observations = outlier_flags[
        outlier_flags["Any_IQR_Outlier"]
    ].copy()

    outlier_observations = outlier_observations.sort_values(
        by=FEATURES,
        ascending=False,
    )

    outlier_observations.head(EXTREME_ROWS).to_csv(
        OUTPUT_DIR / "extreme_observations.csv",
        index=False,
    )

    print("\nOUTLIERS BY CLUSTER")
    print(cluster_outlier_summary.round(4).to_string(index=False))

    print("\nOutlier reports saved:")
    print(feature_summary_file)
    print(cluster_outlier_file)
    print(OUTPUT_DIR / "extreme_observations.csv")

    print(
        "\nNote: IQR outliers are flagged for investigation, "
        "not automatically removed."
    )

    return outlier_flags


# ==========================================
# 7. COMPARE K-MEANS CLUSTER COUNTS
# ==========================================

def compare_cluster_counts(X_scaled):

    print("\nK-MEANS CLUSTER COUNT COMPARISON")
    print("-" * 60)

    rows = []

    for k in CLUSTER_COUNTS_TO_COMPARE:

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        labels = model.fit_predict(X_scaled)

        counts = pd.Series(labels).value_counts()

        silhouette = silhouette_score(
            X_scaled,
            labels,
        )

        db_index = davies_bouldin_score(
            X_scaled,
            labels,
        )

        rows.append({
            "K": k,
            "Silhouette_Score": silhouette,
            "Davies_Bouldin_Index": db_index,
            "Inertia": model.inertia_,
            "Smallest_Cluster": int(counts.min()),
            "Largest_Cluster": int(counts.max()),
        })

        print(
            f"K={k}: Silhouette={silhouette:.4f}, "
            f"DB Index={db_index:.4f}, "
            f"Smallest cluster={counts.min():,}, "
            f"Largest cluster={counts.max():,}"
        )

    comparison_df = pd.DataFrame(rows)

    comparison_file = (
        OUTPUT_DIR / "cluster_count_comparison.csv"
    )

    comparison_df.round(6).to_csv(
        comparison_file,
        index=False,
    )

    print(f"\nComparison saved: {comparison_file}")

    return comparison_df


# ==========================================
# 8. MAIN FUNCTION
# ==========================================

def main():

    print("=" * 65)
    print("RAIN DROP DATA: CLUSTER AND OUTLIER ANALYSIS")
    print("=" * 65)

    df = load_data()

    X_scaled = scale_features(df)

    model, labels = train_model(X_scaled)

    result, profile = analyze_clusters(df, labels)

    investigate_outliers(result)

    comparison_df = compare_cluster_counts(X_scaled)

    print("\nFINAL ANALYSIS SUMMARY")
    print("-" * 65)
    print(f"Observations analyzed: {len(df):,}")
    print(f"Features used: {', '.join(FEATURES)}")
    print(f"Primary model clusters: {N_CLUSTERS}")
    print(f"Primary model inertia: {model.inertia_:.4f}")
    print(
        "Primary model silhouette: "
        f"{silhouette_score(X_scaled, labels):.4f}"
    )
    print(
        "Primary model Davies-Bouldin Index: "
        f"{davies_bouldin_score(X_scaled, labels):.4f}"
    )

    print("\nGenerated reports:")
    print(f"- {OUTPUT_DIR / 'cluster_profiles.csv'}")
    print(f"- {OUTPUT_DIR / 'cluster_assignments.csv'}")
    print(f"- {OUTPUT_DIR / 'iqr_outlier_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'cluster_outlier_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'extreme_observations.csv'}")
    print(f"- {OUTPUT_DIR / 'cluster_count_comparison.csv'}")

    print("\nCluster and outlier analysis completed successfully.")


if __name__ == "__main__":
    main()

