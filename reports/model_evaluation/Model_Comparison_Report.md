# Rain Drop Volume Prediction Using AI

## Model Evaluation and Comparison Report

### 1. Experiment Setup

- Learning approach: Unsupervised learning.
- Primary algorithm: K-Means clustering.
- Comparison algorithm: DBSCAN clustering.
- Sample size: 20,000 observations.
- Features: Diameter_mm and Velocity_m_s.
- Feature scaling: StandardScaler.
- Evaluation metrics: Silhouette Score, Davies-Bouldin Index and cluster sizes.

The reported results are based on a sample of 20,000 observations, not the entire dataset.

### 2. K-Means Results

| Parameters | Clusters | Silhouette Score | Davies-Bouldin Index | Inertia | Smallest Cluster | Largest Cluster |
|---|---:|---:|---:|---:|---:|---:|
| k=2 | 2 | 0.5272 | 0.8863 | 25104.7625 | 5979 | 14021 |
| k=3 | 3 | 0.5322 | 0.6292 | 14067.5567 | 72 | 13935 |
| k=4 | 4 | 0.5041 | 0.7359 | 10430.6828 | 67 | 13185 |
| k=5 | 5 | 0.4140 | 0.7631 | 8145.1479 | 66 | 10185 |
| k=6 | 6 | 0.3840 | 0.7908 | 6869.5740 | 66 | 8305 |
| k=7 | 7 | 0.4071 | 0.7968 | 5844.9647 | 66 | 7659 |
| k=8 | 8 | 0.3738 | 0.7967 | 5129.4846 | 66 | 5469 |

### 3. DBSCAN Results

| Parameters | Clusters | Noise (%) | Silhouette Score | Davies-Bouldin Index |
|---|---:|---:|---:|---:|
| eps=0.1, min_samples=10 | 8 | 4.3600 | 0.0976 | 0.5640 |
| eps=0.15, min_samples=10 | 6 | 2.7450 | 0.2886 | 0.4011 |
| eps=0.2, min_samples=10 | 3 | 1.8050 | 0.6116 | 0.6563 |
| eps=0.25, min_samples=10 | 1 | 1.3300 | N/A | N/A |
| eps=0.3, min_samples=10 | 1 | 1.1450 | N/A | N/A |
| eps=0.4, min_samples=10 | 1 | 0.8850 | N/A | N/A |
| eps=0.5, min_samples=10 | 2 | 0.6200 | 0.7025 | 0.2909 |

### 4. Best K-Means Candidate

- Parameters: k=3
- Number of clusters: 3
- Silhouette Score: 0.5322
- Davies-Bouldin Index: 0.6292
- Smallest cluster: 72 observations
- Largest cluster: 13935 observations

Selection is based on the evaluation metrics available in the sample experiment. Cluster balance and interpretability must also be considered.

### 5. Best DBSCAN Candidate

- Parameters: eps=0.5, min_samples=10
- Number of clusters: 2
- Noise percentage: 0.6200%
- Silhouette Score: 0.7025
- Davies-Bouldin Index: 0.2909

DBSCAN scores may be calculated only on non-noise observations. Therefore, compare them with K-Means scores cautiously.

### 6. Interpretation and Limitations

- A higher Silhouette Score generally indicates better cluster separation.
- A lower Davies-Bouldin Index generally indicates more compact and separated clusters.
- K-Means inertia decreases as the number of clusters increases; it should not be used alone to select k.
- Very small clusters require investigation before assigning them a physical meaning.
- DBSCAN may classify observations as noise.
- These experiments do not establish that clusters represent specific meteorological categories.
- Volume_mm3 is derived from Diameter_mm using the spherical-drop volume formula, so it was not used as an independent clustering feature.
- Results must be validated on additional data before making broader conclusions.

### 7. Next Steps

1. Inspect and interpret the cluster profiles.
2. Validate the chosen configuration on a larger sample.
3. Review the data source, measurement units and meteorological meaning of the observations.
4. Document the final model choice and limitations.

---

Generated automatically from the evaluation CSV files.
