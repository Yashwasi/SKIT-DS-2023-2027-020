from pathlib import Path
import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"

KMEANS_FILE = REPORT_DIR / "kmeans_evaluation.csv"
DBSCAN_FILE = REPORT_DIR / "dbscan_evaluation.csv"

OUTPUT_FILE = REPORT_DIR / "Model_Comparison_Report.md"


# ==========================================
# 2. LOAD EVALUATION RESULTS
# ==========================================

def load_results(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {file_path}"
        )

    return pd.read_csv(file_path)


# ==========================================
# 3. FORMAT NUMBERS
# ==========================================

def format_number(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:.4f}"


# ==========================================
# 4. GENERATE MODEL REPORT
# ==========================================

def generate_report():

    kmeans = load_results(KMEANS_FILE)
    dbscan = load_results(DBSCAN_FILE)

    required_columns = [
        "Parameters",
        "Clusters",
        "Noise_Percent",
        "Silhouette_Score",
        "Davies_Bouldin_Index",
        "Inertia",
        "Smallest_Cluster",
        "Largest_Cluster",
    ]

    for name, df in [
        ("K-Means", kmeans),
        ("DBSCAN", dbscan),
    ]:
        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} CSV is missing columns: {missing}"
            )

    # K-Means candidate: highest silhouette score.
    # This is a candidate selection rule, not automatic proof
    # that the model is physically meaningful.
    valid_kmeans = kmeans.dropna(
        subset=["Silhouette_Score", "Davies_Bouldin_Index"]
    )

    if valid_kmeans.empty:
        raise ValueError("No valid K-Means evaluation rows found.")

    best_kmeans = valid_kmeans.sort_values(
        by=["Silhouette_Score", "Davies_Bouldin_Index"],
        ascending=[False, True],
    ).iloc[0]

    # DBSCAN candidate: highest silhouette among valid rows.
    # DBSCAN metrics may exclude noise observations.
    valid_dbscan = dbscan.dropna(
        subset=["Silhouette_Score", "Davies_Bouldin_Index"]
    )

    if valid_dbscan.empty:
        best_dbscan = None
    else:
        best_dbscan = valid_dbscan.sort_values(
            by=["Silhouette_Score", "Davies_Bouldin_Index"],
            ascending=[False, True],
        ).iloc[0]

    # ==========================================
    # 5. BUILD MARKDOWN REPORT
    # ==========================================

    lines = [
        "# Rain Drop Volume Prediction Using AI",
        "",
        "## Model Evaluation and Comparison Report",
        "",
        "### 1. Experiment Setup",
        "",
        "- Learning approach: Unsupervised learning.",
        "- Primary algorithm: K-Means clustering.",
        "- Comparison algorithm: DBSCAN clustering.",
        "- Sample size: 20,000 observations.",
        "- Features: Diameter_mm and Velocity_m_s.",
        "- Feature scaling: StandardScaler.",
        "- Evaluation metrics: Silhouette Score, "
        "Davies-Bouldin Index and cluster sizes.",
        "",
        "The reported results are based on a sample of "
        "20,000 observations, not the entire dataset.",
        "",
        "### 2. K-Means Results",
        "",
        "| Parameters | Clusters | Silhouette Score | "
        "Davies-Bouldin Index | Inertia | Smallest Cluster | "
        "Largest Cluster |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in kmeans.iterrows():
        lines.append(
            f"| {row['Parameters']} "
            f"| {int(row['Clusters'])} "
            f"| {format_number(row['Silhouette_Score'])} "
            f"| {format_number(row['Davies_Bouldin_Index'])} "
            f"| {format_number(row['Inertia'])} "
            f"| {int(row['Smallest_Cluster'])} "
            f"| {int(row['Largest_Cluster'])} |"
        )

    lines.extend([
        "",
        "### 3. DBSCAN Results",
        "",
        "| Parameters | Clusters | Noise (%) | Silhouette Score | "
        "Davies-Bouldin Index |",
        "|---|---:|---:|---:|---:|",
    ])

    for _, row in dbscan.iterrows():
        lines.append(
            f"| {row['Parameters']} "
            f"| {int(row['Clusters'])} "
            f"| {format_number(row['Noise_Percent'])} "
            f"| {format_number(row['Silhouette_Score'])} "
            f"| {format_number(row['Davies_Bouldin_Index'])} |"
        )

    lines.extend([
        "",
        "### 4. Best K-Means Candidate",
        "",
        f"- Parameters: {best_kmeans['Parameters']}",
        f"- Number of clusters: {int(best_kmeans['Clusters'])}",
        f"- Silhouette Score: "
        f"{format_number(best_kmeans['Silhouette_Score'])}",
        f"- Davies-Bouldin Index: "
        f"{format_number(best_kmeans['Davies_Bouldin_Index'])}",
        f"- Smallest cluster: "
        f"{int(best_kmeans['Smallest_Cluster'])} observations",
        f"- Largest cluster: "
        f"{int(best_kmeans['Largest_Cluster'])} observations",
        "",
        "Selection is based on the evaluation metrics available "
        "in the sample experiment. Cluster balance and "
        "interpretability must also be considered.",
        "",
        "### 5. Best DBSCAN Candidate",
        "",
    ])

    if best_dbscan is not None:
        lines.extend([
            f"- Parameters: {best_dbscan['Parameters']}",
            f"- Number of clusters: {int(best_dbscan['Clusters'])}",
            f"- Noise percentage: "
            f"{format_number(best_dbscan['Noise_Percent'])}%",
            f"- Silhouette Score: "
            f"{format_number(best_dbscan['Silhouette_Score'])}",
            f"- Davies-Bouldin Index: "
            f"{format_number(best_dbscan['Davies_Bouldin_Index'])}",
            "",
            "DBSCAN scores may be calculated only on non-noise "
            "observations. Therefore, compare them with K-Means "
            "scores cautiously.",
        ])
    else:
        lines.append(
            "No DBSCAN row has valid Silhouette and "
            "Davies-Bouldin scores."
        )

    lines.extend([
        "",
        "### 6. Interpretation and Limitations",
        "",
        "- A higher Silhouette Score generally indicates better "
        "cluster separation.",
        "- A lower Davies-Bouldin Index generally indicates "
        "more compact and separated clusters.",
        "- K-Means inertia decreases as the number of clusters "
        "increases; it should not be used alone to select k.",
        "- Very small clusters require investigation before "
        "assigning them a physical meaning.",
        "- DBSCAN may classify observations as noise.",
        "- These experiments do not establish that clusters "
        "represent specific meteorological categories.",
        "- Volume_mm3 is derived from Diameter_mm using the "
        "spherical-drop volume formula, so it was not used as "
        "an independent clustering feature.",
        "- Results must be validated on additional data before "
        "making broader conclusions.",
        "",
        "### 7. Next Steps",
        "",
        "1. Inspect and interpret the cluster profiles.",
        "2. Validate the chosen configuration on a larger sample.",
        "3. Review the data source, measurement units and "
        "meteorological meaning of the observations.",
        "4. Document the final model choice and limitations.",
        "",
        "---",
        "",
        "Generated automatically from the evaluation CSV files.",
        "",
    ])

    OUTPUT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Model comparison report generated successfully.")
    print(f"Report location: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_report()