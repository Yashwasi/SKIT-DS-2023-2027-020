from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"
PLOT_DIR = REPORT_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ==========================================
# 2. K-MEANS EVALUATION VISUALIZATION
# ==========================================

def plot_kmeans_evaluation():

    file_path = REPORT_DIR / "kmeans_evaluation.csv"

    if not file_path.exists():
        print(f"ERROR: K-Means CSV not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    # Validate required columns
    required_columns = [
        "Parameters",
        "Clusters",
        "Silhouette_Score",
        "Davies_Bouldin_Index",
        "Inertia",
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        print(f"ERROR: Missing K-Means columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Extract k values from Parameters, e.g. k=2, k=3
    df["k"] = (
        df["Parameters"]
        .astype(str)
        .str.extract(r"k=(\d+)", expand=False)
    )

    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df = df.dropna(subset=["k"]).sort_values("k")

    if df.empty:
        print("ERROR: No valid K-Means k values found.")
        return

    # Graph 1: Silhouette Score
    plt.figure(figsize=(9, 5))

    sns.lineplot(
        data=df,
        x="k",
        y="Silhouette_Score",
        marker="o",
    )

    plt.title("K-Means: Silhouette Score vs Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.xticks(df["k"].astype(int))
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "kmeans_silhouette.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Graph 2: Davies-Bouldin Index
    plt.figure(figsize=(9, 5))

    sns.lineplot(
        data=df,
        x="k",
        y="Davies_Bouldin_Index",
        marker="o",
    )

    plt.title("K-Means: Davies-Bouldin Index vs Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Davies-Bouldin Index")
    plt.xticks(df["k"].astype(int))
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "kmeans_davies_bouldin.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Graph 3: Elbow Method
    plt.figure(figsize=(9, 5))

    sns.lineplot(
        data=df,
        x="k",
        y="Inertia",
        marker="o",
    )

    plt.title("K-Means: Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(df["k"].astype(int))
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "kmeans_elbow.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("SUCCESS: K-Means graphs saved.")


# ==========================================
# 3. DBSCAN EVALUATION VISUALIZATION
# ==========================================

def plot_dbscan_evaluation():

    file_path = REPORT_DIR / "dbscan_evaluation.csv"

    if not file_path.exists():
        print(f"ERROR: DBSCAN CSV not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    # Validate required columns
    required_columns = [
        "Parameters",
        "Noise_Percent",
        "Silhouette_Score",
        "Davies_Bouldin_Index",
        "Clusters",
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        print(f"ERROR: Missing DBSCAN columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Extract eps from Parameters, e.g. eps=0.1
    df["eps"] = (
        df["Parameters"]
        .astype(str)
        .str.extract(r"eps=([0-9]*\.?[0-9]+)", expand=False)
    )

    df["eps"] = pd.to_numeric(df["eps"], errors="coerce")
    df = df.dropna(subset=["eps"]).sort_values("eps")

    if df.empty:
        print("ERROR: No valid DBSCAN epsilon values found.")
        print("Check the Parameters column in dbscan_evaluation.csv.")
        return

    # Graph 1: Silhouette Score
    # NaN values can occur when DBSCAN produces only one cluster.
    valid_silhouette = df.dropna(subset=["Silhouette_Score"])

    if not valid_silhouette.empty:

        plt.figure(figsize=(9, 5))

        sns.lineplot(
            data=valid_silhouette,
            x="eps",
            y="Silhouette_Score",
            marker="o",
        )

        plt.title("DBSCAN: Silhouette Score vs Epsilon")
        plt.xlabel("Epsilon (eps)")
        plt.ylabel("Silhouette Score")
        plt.tight_layout()

        plt.savefig(
            PLOT_DIR / "dbscan_silhouette.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    else:
        print("WARNING: No valid DBSCAN silhouette scores to plot.")

    # Graph 2: Noise Percentage
    plt.figure(figsize=(9, 5))

    sns.lineplot(
        data=df,
        x="eps",
        y="Noise_Percent",
        marker="o",
    )

    plt.title("DBSCAN: Noise Percentage vs Epsilon")
    plt.xlabel("Epsilon (eps)")
    plt.ylabel("Noise Percentage (%)")
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "dbscan_noise_percentage.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Graph 3: Number of clusters
    plt.figure(figsize=(9, 5))

    sns.lineplot(
        data=df,
        x="eps",
        y="Clusters",
        marker="o",
    )

    plt.title("DBSCAN: Number of Clusters vs Epsilon")
    plt.xlabel("Epsilon (eps)")
    plt.ylabel("Number of Clusters")
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "dbscan_cluster_count.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("SUCCESS: DBSCAN graphs saved.")


# ==========================================
# 4. MAIN FUNCTION
# ==========================================

def main():

    print("=" * 50)
    print("MODEL EVALUATION VISUALIZATION")
    print("=" * 50)

    plot_kmeans_evaluation()
    plot_dbscan_evaluation()

    print("\nVisualization process completed.")
    print(f"Graph folder: {PLOT_DIR}")

    if PLOT_DIR.exists():
        graph_files = list(PLOT_DIR.glob("*.png"))

        print(f"Total PNG graphs available: {len(graph_files)}")

        for graph in sorted(graph_files):
            print(f" - {graph.name}")


if __name__ == "__main__":
    main()