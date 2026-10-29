from pathlib import Path

import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"

CLUSTER_DIR = PROJECT_ROOT / "reports" / "cluster_analysis"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "Final_Model_Validation_Report.md"
)


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_csv(path, required=True):

    if not path.exists():

        if required:
            raise FileNotFoundError(
                f"Required report not found: {path}"
            )

        return None

    return pd.read_csv(path)


def fmt(value, decimals=4):

    if pd.isna(value):
        return "N/A"

    return f"{value:.{decimals}f}"


def markdown_table(df):

    if df is None or df.empty:
        return "_No data available._"

    return df.to_markdown(index=False)


# ==========================================
# 3. LOAD REPORTS
# ==========================================

def main():

    print("Loading project evaluation reports...")

    kmeans = load_csv(
        REPORT_DIR / "kmeans_evaluation.csv"
    )

    dbscan = load_csv(
        REPORT_DIR / "dbscan_evaluation.csv"
    )

    profiles = load_csv(
        CLUSTER_DIR / "cluster_profiles.csv"
    )

    initialization = load_csv(
        CLUSTER_DIR / "cluster_stability_results.csv"
    )

    independent = load_csv(
        CLUSTER_DIR / "independent_sample_validation.csv"
    )

    # ==========================================
    # 4. BUILD FINAL REPORT
    # ==========================================

    sections = []

    sections.append(
        "# Rain Drop Volume Prediction Using AI\n"
        "## Final Model Validation Report\n"
    )

    sections.append(
        "## 1. Project Methodology\n\n"
        "- Learning approach: Unsupervised learning.\n"
        "- Primary algorithm: K-Means.\n"
        "- Comparison algorithm: DBSCAN.\n"
        "- Features: Diameter_mm and Velocity_m_s.\n"
        "- Scaling: StandardScaler.\n"
        "- Evaluation sample size: 20,000 observations "
        "per experiment.\n\n"
        "The experiments use diameter and velocity as clustering "
        "features. Volume is excluded because it is mathematically "
        "derived from diameter using the spherical-drop volume "
        "formula.\n"
    )

    sections.append(
        "## 2. K-Means Evaluation\n\n"
        + markdown_table(kmeans)
    )

    sections.append(
        "## 3. DBSCAN Evaluation\n\n"
        + markdown_table(dbscan)
    )

    sections.append(
        "## 4. Cluster Profiles\n\n"
        + markdown_table(profiles)
    )

    sections.append(
        "## 5. Initialization Stability\n\n"
        + markdown_table(initialization)
        + "\n\n"
        "The Adjusted Rand Index (ARI) compares assignments "
        "from different K-Means initializations on the same "
        "sample. ARI values of 1.0 indicate identical partitions "
        "for the tested runs.\n"
    )

    sections.append(
        "## 6. Independent Sample Validation\n\n"
        + markdown_table(independent)
        + "\n\n"
        "The independent samples produced similar Silhouette "
        "Scores and Davies-Bouldin Index values. However, these "
        "metrics do not establish that the cluster labels have "
        "the same meaning across independently fitted models.\n"
    )

    # ==========================================
    # 5. FINDINGS
    # ==========================================

    findings = [
        "### Main Findings",
        "",
        "- K-Means with three clusters is a candidate configuration "
        "for further investigation.",
        "- On the initial sample, the Silhouette Score was "
        "approximately 0.5322.",
        "- Initialization tests produced ARI = 1.0 on the same "
        "sample, indicating consistent partitions across tested "
        "initializations.",
        "- Independent samples produced Silhouette Scores from "
        "approximately 0.5317 to 0.5406.",
        "- Each independent sample contained a small cluster "
        "with approximately 76–83 observations.",
        "- Cluster sizes are imbalanced, so cluster profiles "
        "must be examined carefully.",
        "",
        "### Limitations",
        "",
        "- Experiments were performed on samples, not on all "
        "12,461,324 valid deduplicated records.",
        "- Cluster IDs are arbitrary labels, not rainfall classes.",
        "- No verified mapping from cluster IDs to light, moderate "
        "or heavy rainfall has been established.",
        "- The meteorological meaning and provenance of the "
        "observations require confirmation.",
        "- Silhouette Score and Davies-Bouldin Index evaluate "
        "geometric clustering quality, not rainfall prediction "
        "accuracy.",
        "- A high initialization ARI does not guarantee "
        "generalization to new data.",
        "",
        "### Conclusion",
        "",
        "K-Means with three clusters is a promising candidate "
        "for this sample-based unsupervised clustering experiment. "
        "The observed metrics are relatively consistent across "
        "independent samples, but the cluster imbalance and "
        "meteorological interpretation remain unresolved. "
        "Further validation is required before selecting a final "
        "production model or assigning physical rainfall labels.",
        "",
        "### Recommended Next Steps",
        "",
        "1. Investigate the small cluster and its extreme values.",
        "2. Compare cluster profiles with verified domain knowledge.",
        "3. Evaluate computational feasibility on a larger sample.",
        "4. Confirm dataset provenance, units and measurement context.",
        "5. If rainfall forecasting is required, obtain suitable "
        "time-indexed meteorological data and define a target variable.",
    ]

    sections.append("\n".join(findings))

    # ==========================================
    # 6. SAVE REPORT
    # ==========================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        "\n\n".join(sections),
        encoding="utf-8",
    )

    print("\nFinal validation report generated successfully.")
    print(f"Report location: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()