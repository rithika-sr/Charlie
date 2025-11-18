# ml_src/api/app.py

from __future__ import annotations

import os
import joblib
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from ml_src.utils.logging import get_logger

logger = get_logger("api")

# Path to the model inside the container / repo
MODEL_PATH = os.getenv("MODEL_PATH", "models/final_model.joblib")

app = FastAPI(
    title="Charlie MBTA Delay Model API",
    description=(
        "FastAPI service exposing the trained MBTA delay prediction model. "
        "Takes direction_id and stop_sequence and returns delay probability."
    ),
    version="1.0.0",
)

model = None


class PredictionRequest(BaseModel):
    direction_id: int
    stop_sequence: int


class PredictionResponse(BaseModel):
    prob_delayed: float
    label: int
    model_version: Optional[str] = None


def load_model():
    """Load model from disk once at startup."""
    global model

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        logger.error(f"❌ Model file not found at {model_path.resolve()}")
        raise FileNotFoundError(f"Model file not found at {model_path}")

    logger.info(f"📦 Loading model from {model_path.resolve()}")
    model = joblib.load(model_path)
    logger.info("✅ Model loaded successfully")


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health", tags=["health"])
def health_check():
    """Simple health endpoint for Cloud Run / monitoring."""
    exists = Path(MODEL_PATH).exists()
    return {
        "status": "ok" if exists else "model_missing",
        "model_path": str(Path(MODEL_PATH).resolve()),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(req: PredictionRequest):
    """
    Predict delay given MBTA trip features.

    Currently uses:
    - direction_id (0 / 1)
    - stop_sequence (integer)
    """
    if model is None:
        # Should not happen if startup_event succeeds
        load_model()

    # Build a 2D array for sklearn-style predict
    X = [[req.direction_id, req.stop_sequence]]

    # Predict probability & label
    try:
        prob = float(model.predict_proba(X)[0][1])
    except Exception:
        # If model doesn't support predict_proba (but yours does)
        preds = model.predict(X)
        prob = float(preds[0])

    label = int(prob >= 0.5)

    # Optional: model version from env (you can set this in CI)
    model_version = os.getenv("MODEL_VERSION", "local")

    return PredictionResponse(
        prob_delayed=prob,
        label=label,
        model_version=model_version,
    )