# Rain Drop Volume Prediction Using AI
## Final Model Validation Report


## 1. Project Methodology

- Learning approach: Unsupervised learning.
- Primary algorithm: K-Means.
- Comparison algorithm: DBSCAN.
- Features: Diameter_mm and Velocity_m_s.
- Scaling: StandardScaler.
- Evaluation sample size: 20,000 observations per experiment.

The experiments use diameter and velocity as clustering features. Volume is excluded because it is mathematically derived from diameter using the spherical-drop volume formula.


## 2. K-Means Evaluation

| Algorithm   | Parameters   |   Clusters |   Noise_Percent |   Silhouette_Score |   Davies_Bouldin_Index |   Inertia |   Smallest_Cluster |   Largest_Cluster |
|:------------|:-------------|-----------:|----------------:|-------------------:|-----------------------:|----------:|-------------------:|------------------:|
| KMeans      | k=2          |          2 |               0 |           0.527192 |               0.886325 |  25104.8  |               5979 |             14021 |
| KMeans      | k=3          |          3 |               0 |           0.532221 |               0.629222 |  14067.6  |                 72 |             13935 |
| KMeans      | k=4          |          4 |               0 |           0.504079 |               0.735876 |  10430.7  |                 67 |             13185 |
| KMeans      | k=5          |          5 |               0 |           0.414032 |               0.763085 |   8145.15 |                 66 |             10185 |
| KMeans      | k=6          |          6 |               0 |           0.384042 |               0.790828 |   6869.57 |                 66 |              8305 |
| KMeans      | k=7          |          7 |               0 |           0.407103 |               0.796846 |   5844.96 |                 66 |              7659 |
| KMeans      | k=8          |          8 |               0 |           0.373848 |               0.796695 |   5129.48 |                 66 |              5469 |

## 3. DBSCAN Evaluation

| Algorithm   | Parameters               |   Clusters |   Noise_Percent |   Silhouette_Score |   Davies_Bouldin_Index |   Inertia |   Smallest_Cluster |   Largest_Cluster |
|:------------|:-------------------------|-----------:|----------------:|-------------------:|-----------------------:|----------:|-------------------:|------------------:|
| DBSCAN      | eps=0.1, min_samples=10  |          8 |           4.36  |          0.0975997 |               0.564049 |       nan |                  8 |             18994 |
| DBSCAN      | eps=0.15, min_samples=10 |          6 |           2.745 |          0.288634  |               0.401099 |       nan |                  7 |             19389 |
| DBSCAN      | eps=0.2, min_samples=10  |          3 |           1.805 |          0.611606  |               0.656278 |       nan |                 13 |             19590 |
| DBSCAN      | eps=0.25, min_samples=10 |          1 |           1.33  |        nan         |             nan        |       nan |              19734 |             19734 |
| DBSCAN      | eps=0.3, min_samples=10  |          1 |           1.145 |        nan         |             nan        |       nan |              19771 |             19771 |
| DBSCAN      | eps=0.4, min_samples=10  |          1 |           0.885 |        nan         |             nan        |       nan |              19823 |             19823 |
| DBSCAN      | eps=0.5, min_samples=10  |          2 |           0.62  |          0.702495  |               0.290862 |       nan |                 15 |             19861 |

## 4. Cluster Profiles

|   Cluster |   Count |   Diameter_Mean |   Diameter_Median |   Diameter_Min |   Diameter_Max |   Velocity_Mean |   Velocity_Median |   Velocity_Min |   Velocity_Max |   Percentage |
|----------:|--------:|----------------:|------------------:|---------------:|---------------:|----------------:|------------------:|---------------:|---------------:|-------------:|
|         0 |   13935 |          0.7151 |            0.687  |          0.521 |          2.626 |          1.7329 |             1.708 |          1     |          2.725 |       69.675 |
|         1 |    5993 |          0.8216 |            0.765  |          0.52  |          2.483 |          3.5319 |             3.422 |          1.718 |         10.223 |       29.965 |
|         2 |      72 |          4.5526 |            4.7545 |          2.676 |          6.016 |          2.6892 |             2.59  |          1.292 |          8.581 |        0.36  |

## 5. Initialization Stability

|   Random_State |   Silhouette_Score |   Davies_Bouldin_Index |   Inertia |   Smallest_Cluster |   Largest_Cluster |   ARI_vs_Seed_0 |
|---------------:|-------------------:|-----------------------:|----------:|-------------------:|------------------:|----------------:|
|              0 |           0.532221 |               0.629222 |   14067.6 |                 72 |             13935 |               1 |
|             21 |           0.532221 |               0.629222 |   14067.6 |                 72 |             13935 |               1 |
|             42 |           0.532221 |               0.629222 |   14067.6 |                 72 |             13935 |               1 |
|            100 |           0.532221 |               0.629222 |   14067.6 |                 72 |             13935 |               1 |
|           2026 |           0.532221 |               0.629222 |   14067.6 |                 72 |             13935 |               1 |

The Adjusted Rand Index (ARI) compares assignments from different K-Means initializations on the same sample. ARI values of 1.0 indicate identical partitions for the tested runs.


## 6. Independent Sample Validation

|   Sample_Seed |   Observations |   Silhouette_Score |   Davies_Bouldin_Index |   Inertia |   Smallest_Cluster |   Largest_Cluster |
|--------------:|---------------:|-------------------:|-----------------------:|----------:|-------------------:|------------------:|
|            42 |          20000 |           0.531659 |               0.661346 |   14908.4 |                 77 |             14008 |
|           100 |          20000 |           0.536392 |               0.645467 |   14744.4 |                 76 |             13974 |
|          2026 |          20000 |           0.540576 |               0.631196 |   13898.4 |                 83 |             14150 |

The independent samples produced similar Silhouette Scores and Davies-Bouldin Index values. However, these metrics do not establish that the cluster labels have the same meaning across independently fitted models.


### Main Findings

- K-Means with three clusters is a candidate configuration for further investigation.
- On the initial sample, the Silhouette Score was approximately 0.5322.
- Initialization tests produced ARI = 1.0 on the same sample, indicating consistent partitions across tested initializations.
- Independent samples produced Silhouette Scores from approximately 0.5317 to 0.5406.
- Each independent sample contained a small cluster with approximately 76–83 observations.
- Cluster sizes are imbalanced, so cluster profiles must be examined carefully.

### Limitations

- Experiments were performed on samples, not on all 12,461,324 valid deduplicated records.
- Cluster IDs are arbitrary labels, not rainfall classes.
- No verified mapping from cluster IDs to light, moderate or heavy rainfall has been established.
- The meteorological meaning and provenance of the observations require confirmation.
- Silhouette Score and Davies-Bouldin Index evaluate geometric clustering quality, not rainfall prediction accuracy.
- A high initialization ARI does not guarantee generalization to new data.

### Conclusion

K-Means with three clusters is a promising candidate for this sample-based unsupervised clustering experiment. The observed metrics are relatively consistent across independent samples, but the cluster imbalance and meteorological interpretation remain unresolved. Further validation is required before selecting a final production model or assigning physical rainfall labels.

### Recommended Next Steps

1. Investigate the small cluster and its extreme values.
2. Compare cluster profiles with verified domain knowledge.
3. Evaluate computational feasibility on a larger sample.
4. Confirm dataset provenance, units and measurement context.
5. If rainfall forecasting is required, obtain suitable time-indexed meteorological data and define a target variable.