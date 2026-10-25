from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "raindrop_volume_processed"
    / "raindrop_volume_deduplicated.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 20_000

FEATURE_COLUMNS = ["Diameter_mm", "Velocity_m_s"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print("Loading dataset sample...")

    df = pd.read_csv(
        DATA_PATH,
        usecols=FEATURE_COLUMNS,
        nrows=SAMPLE_SIZE,
    )

    if df.empty:
        raise ValueError("The dataset sample is empty.")

    if df.isnull().any().any():
        raise ValueError(
            "Missing values detected in selected features."
        )

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(
            "Non-finite values detected in selected features."
        )

    if (df <= 0).any().any():
        raise ValueError(
            "Non-positive values detected; log1p experiment "
            "requires valid finite measurements."
        )

    print(f"Rows inspected: {len(df):,}")
    print("Selected features:", FEATURE_COLUMNS)

    # Approach A: original features
    X_original = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    scaler_original = StandardScaler()
    X_original_scaled = scaler_original.fit_transform(X_original)

    original_output = pd.DataFrame(
        X_original_scaled,
        columns=[
            "Diameter_original_scaled",
            "Velocity_original_scaled",
        ],
    )

    # Approach B: log-transformed features
    X_log = np.log1p(X_original)

    scaler_log = StandardScaler()
    X_log_scaled = scaler_log.fit_transform(X_log)

    log_output = pd.DataFrame(
        X_log_scaled,
        columns=[
            "Diameter_log1p_scaled",
            "Velocity_log1p_scaled",
        ],
    )

    # Keep original feature values alongside the transformed versions
    output = pd.concat(
        [
            df.reset_index(drop=True),
            original_output,
            log_output,
        ],
        axis=1,
    )

    output_path = OUTPUT_DIR / "feature_engineering_sample.csv"
    output.to_csv(output_path, index=False)

    print("\nFeature engineering summary")
    print("=" * 55)
    print("Output rows:", len(output))
    print("Output columns:", list(output.columns))
    print("Output file:", output_path)

    print("\nStandardized feature means:")
    print(
        output[
            [
                "Diameter_original_scaled",
                "Velocity_original_scaled",
                "Diameter_log1p_scaled",
                "Velocity_log1p_scaled",
            ]
        ].mean().round(4)
    )

    print("\nFeature engineering sample completed successfully.")


if __name__ == "__main__":
    main()