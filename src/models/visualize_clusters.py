from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "reports"
    / "cluster_analysis"
    / "cluster_assignments.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "cluster_analysis"
    / "plots"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD CLUSTER ASSIGNMENTS
# ==========================================

def load_cluster_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cluster assignments not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "Diameter_mm",
        "Velocity_m_s",
        "Cluster",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print(f"Loaded observations: {len(df):,}")

    return df


# ==========================================
# 3. CREATE SCATTER PLOT
# ==========================================

def plot_clusters(df):

    plt.figure(figsize=(11, 7))

    for cluster_id in sorted(df["Cluster"].unique()):

        cluster_data = df[df["Cluster"] == cluster_id]

        plt.scatter(
            cluster_data["Diameter_mm"],
            cluster_data["Velocity_m_s"],
            label=(
                f"Cluster {cluster_id} "
                f"(n={len(cluster_data):,})"
            ),
            alpha=0.6,
            s=18,
        )

    plt.title(
        "K-Means Cluster Analysis: "
        "Drop Diameter vs Velocity"
    )

    plt.xlabel("Diameter (mm)")
    plt.ylabel("Velocity (m/s)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = OUTPUT_DIR / "cluster_scatter.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Scatter plot saved: {output_file}")


# ==========================================
# 4. CLUSTER SIZE CHART
# ==========================================

def plot_cluster_sizes(df):

    counts = (
        df["Cluster"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        counts.index.astype(str),
        counts.values,
    )

    plt.title("Number of Observations per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Observations")

    plt.bar_label(bars, fmt="%d")

    plt.tight_layout()

    output_file = OUTPUT_DIR / "cluster_sizes.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Cluster size chart saved: {output_file}")


# ==========================================
# 5. MAIN
# ==========================================

def main():

    print("=" * 55)
    print("CLUSTER VISUALIZATION")
    print("=" * 55)

    df = load_cluster_data()

    plot_clusters(df)
    plot_cluster_sizes(df)

    print("\nAll cluster plots generated successfully.")


if __name__ == "__main__":
    main()