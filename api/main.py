"""
api/main.py
-----------
User Story: To Develop REST APIs (01/11/2026 – 30/11/2026)
Purpose   : FastAPI backend exposing REST endpoints for the
            Rain Drop Volume Prediction system.

Endpoints
---------
GET  /                        – Health check / project info
GET  /health                  – Liveness probe
POST /predict/volume          – Predict Volume_mm3 from Diameter_mm & Velocity_m_s
POST /predict/batch           – Batch prediction (list of samples)
GET  /models/info             – Loaded model metadata
POST /validate                – Validate a dataset row (data quality check)
GET  /dataset/stats           – Summary statistics of the loaded dataset sample

Run locally:
    uvicorn api.main:app --reload --port 8000

Or from the project root:
    python -m uvicorn api.main:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import math
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

# ── Optional data / ML dependencies (lazy so server starts without them) ──
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    xgb = None  # type: ignore[assignment]
    _HAS_XGB = False

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False

if TYPE_CHECKING:
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    import torch  # noqa: F811
    import torch.nn as nn  # noqa: F811
    import xgboost as xgb  # noqa: F811

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent

MODEL_DIR = PROJECT_ROOT / "models"
XGB_MODEL_PATH = MODEL_DIR / "xgboost_volume_predictor.json"
NN_MODEL_PATH = MODEL_DIR / "pytorch_raindrop_net.pt"

DATA_CANDIDATES = [
    PROJECT_ROOT
    / "raindrop_volume_processed"
    / "raindrop_volume_clean.csv",
    PROJECT_ROOT
    / "raindrop_volume_processed"
    / "raindrop_volume_deduplicated.csv",
]

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = ["Diameter_mm", "Velocity_m_s"]
HIDDEN_DIMS = [64, 64, 32]
STARTUP_TIME = time.time()


# ---------------------------------------------------------------------------
# PYTORCH MODEL DEFINITION
# ---------------------------------------------------------------------------

_NetBase: Any = nn.Module if _HAS_TORCH else object

class RaindropNet(_NetBase):  # type: ignore[misc,valid-type]
    """Fully-connected regression network (must match train_models.py)."""

    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# MODEL / SCALER CACHE
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_xgb():
    if not _HAS_XGB:
        return None
    if not XGB_MODEL_PATH.exists():
        return None
    model = xgb.XGBRegressor()
    model.load_model(str(XGB_MODEL_PATH))
    return model


@lru_cache(maxsize=1)
def _load_nn():
    if not _HAS_TORCH:
        return None
    if not NN_MODEL_PATH.exists():
        return None
    net = RaindropNet(input_dim=2, hidden_dims=HIDDEN_DIMS)
    net.load_state_dict(
        torch.load(str(NN_MODEL_PATH), map_location="cpu", weights_only=True)
    )
    net.eval()
    return net


@lru_cache(maxsize=1)
def _load_scaler():
    """Fit a StandardScaler on a small sample — only used for the NN."""
    if not _HAS_NUMPY or not _HAS_PANDAS:
        return None
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    data_path = next(
        (p for p in DATA_CANDIDATES if p.exists()), None
    )
    if data_path is None:
        return None

    df = pd.read_csv(data_path, usecols=FEATURE_COLUMNS, nrows=30_000)
    df = df.dropna()
    df = df[(df > 0).all(axis=1)]
    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    return scaler


@lru_cache(maxsize=1)
def _dataset_stats() -> dict:
    if not _HAS_PANDAS:
        return {}
    data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data_path is None:
        return {}

    df = pd.read_csv(
        data_path,
        usecols=["Diameter_mm", "Velocity_m_s", "Volume_mm3"],
        nrows=30_000,
    )
    df = df.dropna()
    df = df[(df > 0).all(axis=1)]

    stats: dict = {}
    for col in ["Diameter_mm", "Velocity_m_s", "Volume_mm3"]:
        stats[col] = {
            "count": int(df[col].count()),
            "mean": round(float(df[col].mean()), 6),
            "std": round(float(df[col].std()), 6),
            "min": round(float(df[col].min()), 6),
            "max": round(float(df[col].max()), 6),
            "q25": round(float(df[col].quantile(0.25)), 6),
            "median": round(float(df[col].median()), 6),
            "q75": round(float(df[col].quantile(0.75)), 6),
        }
    return stats


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _spherical_volume(diameter_mm: float) -> float:
    return (math.pi / 6.0) * diameter_mm ** 3


def _predict_xgb(diameter: float, velocity: float) -> Optional[float]:
    if not _HAS_NUMPY:
        return None
    model = _load_xgb()
    if model is None:
        return None
    X = np.array([[diameter, velocity]], dtype=np.float64)
    return float(model.predict(X)[0])


def _predict_nn(diameter: float, velocity: float) -> Optional[float]:
    if not _HAS_NUMPY or not _HAS_TORCH:
        return None
    net = _load_nn()
    scaler = _load_scaler()
    if net is None or scaler is None:
        return None
    X = np.array([[diameter, velocity]], dtype=np.float64)
    X_scaled = scaler.transform(X).astype(np.float32)
    X_t = torch.tensor(X_scaled)
    with torch.no_grad():
        return float(net(X_t).item())


# ---------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# ---------------------------------------------------------------------------

class RaindropSample(BaseModel):
    diameter_mm: float = Field(
        ..., gt=0, description="Raindrop diameter in millimetres (must be > 0)"
    )
    velocity_m_s: float = Field(
        ..., gt=0, description="Raindrop velocity in metres per second (must be > 0)"
    )

    @model_validator(mode="after")
    def check_physical_bounds(self) -> "RaindropSample":
        if self.diameter_mm > 10.0:
            raise ValueError(
                "diameter_mm exceeds typical raindrop range (> 10 mm)."
            )
        if self.velocity_m_s > 20.0:
            raise ValueError(
                "velocity_m_s exceeds typical raindrop range (> 20 m/s)."
            )
        return self


class PredictionResponse(BaseModel):
    diameter_mm: float
    velocity_m_s: float
    volume_formula_mm3: float = Field(
        description="Volume computed from the spherical-drop formula V=(π/6)×d³"
    )
    volume_xgboost_mm3: Optional[float] = Field(
        default=None, description="Volume predicted by the XGBoost model"
    )
    volume_pytorch_mm3: Optional[float] = Field(
        default=None, description="Volume predicted by the PyTorch neural network"
    )
    models_available: list[str]


class BatchRequest(BaseModel):
    samples: list[RaindropSample] = Field(
        ..., min_length=1, max_length=1000,
        description="List of raindrop samples (max 1000)"
    )


class BatchPredictionItem(BaseModel):
    diameter_mm: float
    velocity_m_s: float
    volume_formula_mm3: float
    volume_xgboost_mm3: Optional[float]
    volume_pytorch_mm3: Optional[float]


class BatchResponse(BaseModel):
    count: int
    predictions: list[BatchPredictionItem]


class ValidationRequest(BaseModel):
    diameter_mm: Optional[float] = None
    velocity_m_s: Optional[float] = None
    volume_mm3: Optional[float] = None
    condition: Optional[str] = None
    source_file: Optional[str] = None


class ValidationResponse(BaseModel):
    is_valid: bool
    issues: list[str]
    formula_volume_mm3: Optional[float]
    volume_error_mm3: Optional[float]


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Rain Drop Volume Prediction API",
    description=(
        "REST API for the Rain Drop Volume Prediction Using AI project "
        "(SKIT-DS-2023-2027-020). Provides volume prediction endpoints "
        "backed by XGBoost and a PyTorch neural network."
    ),
    version="1.0.0",
    contact={
        "name": "SKIT DS Team",
        "url": "https://github.com/Yashwasi/SKIT-DS-2023-2027-020",
    },
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
async def root():
    """Project overview and API status."""
    return {
        "project": "Rain Drop Volume Prediction Using AI",
        "team": "SKIT-DS-2023-2027-020",
        "version": "1.0.0",
        "status": "running",
        "uptime_seconds": round(time.time() - STARTUP_TIME, 1),
        "docs_url": "/docs",
    }


@app.get("/health", tags=["Info"])
async def health():
    """Liveness probe for deployment / Docker health checks."""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/models/info", tags=["Models"])
async def models_info():
    """Return metadata about loaded models."""
    xgb_available = _load_xgb() is not None
    nn_available = _load_nn() is not None
    scaler_available = _load_scaler() is not None

    data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)

    return {
        "xgboost": {
            "available": xgb_available,
            "path": str(XGB_MODEL_PATH) if XGB_MODEL_PATH.exists() else None,
        },
        "pytorch_nn": {
            "available": nn_available,
            "architecture": f"input→{'→'.join(str(h) for h in HIDDEN_DIMS)}→1 (ReLU)",
            "path": str(NN_MODEL_PATH) if NN_MODEL_PATH.exists() else None,
        },
        "scaler_fitted": scaler_available,
        "dataset_source": str(data_path) if data_path else None,
        "features": FEATURE_COLUMNS,
        "target": "Volume_mm3",
    }


@app.post(
    "/predict/volume",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
async def predict_volume(sample: RaindropSample):
    """
    Predict raindrop Volume_mm3 from Diameter_mm and Velocity_m_s.

    Returns the theoretical formula value alongside XGBoost and
    PyTorch model predictions (when models are available).
    """
    formula_vol = _spherical_volume(sample.diameter_mm)
    xgb_vol = _predict_xgb(sample.diameter_mm, sample.velocity_m_s)
    nn_vol = _predict_nn(sample.diameter_mm, sample.velocity_m_s)

    available = ["formula"]
    if xgb_vol is not None:
        available.append("xgboost")
    if nn_vol is not None:
        available.append("pytorch_nn")

    return PredictionResponse(
        diameter_mm=sample.diameter_mm,
        velocity_m_s=sample.velocity_m_s,
        volume_formula_mm3=round(formula_vol, 8),
        volume_xgboost_mm3=round(xgb_vol, 8) if xgb_vol is not None else None,
        volume_pytorch_mm3=round(nn_vol, 8) if nn_vol is not None else None,
        models_available=available,
    )


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    tags=["Prediction"],
)
async def predict_batch(request: BatchRequest):
    """
    Predict Volume_mm3 for a list of raindrop samples (max 1,000).
    """
    predictions: list[BatchPredictionItem] = []

    for s in request.samples:
        fv = _spherical_volume(s.diameter_mm)
        xv = _predict_xgb(s.diameter_mm, s.velocity_m_s)
        nv = _predict_nn(s.diameter_mm, s.velocity_m_s)

        predictions.append(
            BatchPredictionItem(
                diameter_mm=s.diameter_mm,
                velocity_m_s=s.velocity_m_s,
                volume_formula_mm3=round(fv, 8),
                volume_xgboost_mm3=round(xv, 8) if xv is not None else None,
                volume_pytorch_mm3=round(nv, 8) if nv is not None else None,
            )
        )

    return BatchResponse(count=len(predictions), predictions=predictions)


@app.post(
    "/validate",
    response_model=ValidationResponse,
    tags=["Data Quality"],
)
async def validate_row(row: ValidationRequest):
    """
    Validate a single data row for data-quality issues:
    - Checks for NULL / non-positive physical measurements.
    - Verifies Volume_mm3 against the spherical formula.
    """
    issues: list[str] = []
    formula_volume: Optional[float] = None
    volume_error: Optional[float] = None

    # Check physical measurements
    for field, value in [
        ("diameter_mm", row.diameter_mm),
        ("velocity_m_s", row.velocity_m_s),
        ("volume_mm3", row.volume_mm3),
    ]:
        if value is None:
            issues.append(f"'{field}' is missing (NULL).")
        elif value <= 0:
            issues.append(f"'{field}' must be positive; got {value}.")

    # Volume formula check
    if row.diameter_mm is not None and row.diameter_mm > 0:
        formula_volume = round(_spherical_volume(row.diameter_mm), 8)

        if row.volume_mm3 is not None and row.volume_mm3 > 0:
            volume_error = round(abs(row.volume_mm3 - formula_volume), 8)
            if volume_error > 0.001:
                issues.append(
                    f"volume_mm3 ({row.volume_mm3}) deviates from formula "
                    f"({formula_volume}) by {volume_error:.6f} mm³."
                )

    return ValidationResponse(
        is_valid=len(issues) == 0,
        issues=issues,
        formula_volume_mm3=formula_volume,
        volume_error_mm3=volume_error,
    )


@app.get("/dataset/stats", tags=["Data"])
async def dataset_stats():
    """
    Return descriptive statistics (count, mean, std, min/max, quartiles)
    computed from the first 30,000 rows of the working dataset.
    """
    stats = _dataset_stats()
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Dataset not available. "
                "Ensure raindrop_volume_deduplicated.csv exists."
            ),
        )
    return {"sample_rows": 30_000, "columns": stats}
