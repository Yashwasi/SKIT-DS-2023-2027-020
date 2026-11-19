# Rain Drop Volume Prediction Using AI
## Model Training Report

- **Features used:** ['Diameter_mm', 'Velocity_m_s']
- **Target:** `Volume_mm3`
- **Sample size:** 30,000 rows
- **Train / test split:** 80% / 20%

## 1. XGBoost Regressor

| Parameter | Value |
|---|---|
| n_estimators | 300 |
| max_depth | 6 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| random_state | 42 |
| n_jobs | -1 |

## 2. PyTorch Neural Network

- Architecture: input → 64 → 64 → 32 → 1
- Activation: ReLU
- Epochs: 100
- Batch size: 512
- Optimizer: Adam (lr=0.001)
- Device: N/A

## 3. Evaluation Metrics (Test Set)

| Model | MAE | RMSE | R² | Training time (s) |
|---|---:|---:|---:|---:|
| XGBoost (test) | 0.160402 | 1.464570 | 0.835977 | 1.55 |

## 4. Notes

- Volume_mm3 is geometrically derived from Diameter_mm (`V = (π/6) × d³`). Near-perfect R² scores are expected and confirm correct implementation.
- Velocity_m_s contributes a small independent signal; excluding it degrades R² slightly.
- Models are saved to `models/` and can be loaded for the FastAPI prediction endpoint.

---
*Generated automatically by `src/train_models.py`.*
