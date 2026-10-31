
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "data" / "raindrop_volume_processed.zip"
OUTPUT_DIR = ROOT / "reports" / "model_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 10_000
RANDOM_SEEDS = [42, 123, 2024, 7, 99]
K_VALUES = [2, 3, 4]

def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {ZIP_PATH}")

    print("Reading dataset from ZIP...")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        csv_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]
        if not csv_files:
            raise ValueError("No CSV file found inside dataset ZIP.")

        with archive.open(csv_files[0]) as file:
            df = pd.read_csv(
                file,
                usecols=["Diameter_mm", "Velocity_m_s"],
            )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Diameter_mm", "Velocity_m_s"])
    df = df[
        (df["Diameter_mm"] > 0) &
        (df["Velocity_m_s"] > 0)
    ].reset_index(drop=True)

    if len(df) < SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(df)} valid rows; need {SAMPLE_SIZE}."
        )

    results = []

    for seed in RANDOM_SEEDS:
        sample = df.sample(
            n=SAMPLE_SIZE,
            random_state=seed,
        )

        X = sample[
            ["Diameter_mm", "Velocity_m_s"]
        ].to_numpy(dtype=float)

        X_scaled = StandardScaler().fit_transform(X)

        for k in K_VALUES:
            model = KMeans(
                n_clusters=k,
                n_init=10,
                random_state=42,
            )
            labels = model.fit_predict(X_scaled)

            score = silhouette_score(
                X_scaled,
                labels,
                sample_size=min(3000, len(X_scaled)),
                random_state=seed,
            )

            proportions = (
                pd.Series(labels)
                .value_counts(normalize=True)
                .sort_index()
            )

            row = {
                "Sample_Seed": seed,
                "Sample_Size": len(sample),
                "K": k,
                "Silhouette_Score": score,
                "Inertia": model.inertia_,
                "Smallest_Cluster_Pct": proportions.min() * 100,
                "Largest_Cluster_Pct": proportions.max() * 100,
            }
            results.append(row)

            print(
                f"Seed={seed}, K={k}, "
                f"Silhouette={score:.4f}"
            )

    result_df = pd.DataFrame(results)
    output_file = (
        OUTPUT_DIR / "clustering_stability_results.csv"
    )
    result_df.to_csv(output_file, index=False)

    summary = (
        result_df.groupby("K")
        .agg(
            Mean_Silhouette=("Silhouette_Score", "mean"),
            Std_Silhouette=("Silhouette_Score", "std"),
            Mean_Inertia=("Inertia", "mean"),
            Mean_Smallest_Cluster_Pct=(
                "Smallest_Cluster_Pct", "mean"
            ),
        )
        .reset_index()
    )

    summary_file = (
        OUTPUT_DIR / "clustering_stability_summary.csv"
    )
    summary.to_csv(summary_file, index=False)

    print("\nSUMMARY")
    print(summary.round(4).to_string(index=False))
    print(f"\nDetailed results: {output_file}")
    print(f"Summary results: {summary_file}")

if __name__ == "__main__":
    main()
