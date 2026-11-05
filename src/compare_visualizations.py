
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "models" / "model_comparison_predictions.csv"
OUTPUT_DIR = ROOT / "models"

df = __import__("pandas").read_csv(DATA_PATH)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plots = [
    ("KMeans_Cluster", "K-Means (K = 3)"),
    ("DBSCAN_Cluster", "DBSCAN (eps = 0.5)")
]

for ax, (column, title) in zip(axes, plots):
    scatter = ax.scatter(
        df["Diameter_mm"],
        df["Velocity_m_s"],
        c=df[column],
        s=5,
        alpha=0.45
    )
    ax.set_title(title)
    ax.set_xlabel("Diameter (mm)")
    ax.set_ylabel("Velocity (m/s)")
    fig.colorbar(scatter, ax=ax, label="Cluster (-1 = noise)")

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "model_comparison_visualization.png",
    dpi=150
)
plt.close(fig)

print("Saved: models/model_comparison_visualization.png")
