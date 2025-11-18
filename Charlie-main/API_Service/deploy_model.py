# ml_src/deploy_model.py
"""
Deployment-ready model interface for the MBTA delay classifier.

- Exposes a FastAPI app with a /predict endpoint
- Loads the currently selected production model from models/final_model.joblib
- Can also be run as a CLI smoke test: `python -m ml_src.deploy_model`

To run the API locally after installing `fastapi` and `uvicorn`:

    uvicorn ml_src.deploy_model:app --reload

"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ml_src.utils.logging import get_logger

logger = get_logger("deploy_model")

# Prefer the selected "production" model
FINAL_MODEL_PATH = Path("models/final_model.joblib")
FALLBACK_MODEL_PATH = Path("models/model_lgbm.joblib")


# ----------------------------------------------------
# Pydantic request/response schemas
# ----------------------------------------------------
class PredictionRequest(BaseModel):
    """
    Minimal feature set used by our models.
    """
    direction_id: int
    stop_sequence: int


class PredictionResponse(BaseModel):
    delayed_probability: float
    delayed_prediction: int
    model_name: str


# ----------------------------------------------------
# Model loading helper
# ----------------------------------------------------
def load_model():
    """
    Load the production model for inference.
    Priority:
        1) models/final_model.joblib (selected by model_select.py)
        2) models/model_lgbm.joblib (baseline)
    """
    if FINAL_MODEL_PATH.exists():
        model_path = FINAL_MODEL_PATH
        model_name = "final_model"
    elif FALLBACK_MODEL_PATH.exists():
        model_path = FALLBACK_MODEL_PATH
        model_name = "model_lgbm"
    else:
        raise FileNotFoundError(
            "No model found. Expected one of: "
            f"{FINAL_MODEL_PATH} or {FALLBACK_MODEL_PATH}. "
            "Run model_train.py and model_select.py first."
        )

    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    return model, model_name


# Load at module import so API is ready
MODEL, MODEL_NAME = load_model()


def predict_one(direction_id: int, stop_sequence: int) -> PredictionResponse:
    """
    Core prediction function used by both the API and CLI.
    """
    # Build a single-row dataframe
    X = pd.DataFrame(
        {
            "direction_id": [direction_id],
            "stop_sequence": [stop_sequence],
        }
    )

    # Some sklearn models expose predict_proba; if not, we approximate
    if hasattr(MODEL, "predict_proba"):
        prob = float(MODEL.predict_proba(X)[:, 1][0])
    else:
        # Fallback: use decision_function or prediction as proxy
        if hasattr(MODEL, "decision_function"):
            raw = float(MODEL.decision_function(X)[0])
            # simple logistic transform to keep it between 0 and 1
            import math

            prob = 1 / (1 + math.exp(-raw))
        else:
            pred = int(MODEL.predict(X)[0])
            prob = float(pred)

    pred_label = int(MODEL.predict(X)[0])

    return PredictionResponse(
        delayed_probability=prob,
        delayed_prediction=pred_label,
        model_name=MODEL_NAME,
    )


# ----------------------------------------------------
# FastAPI app
# ----------------------------------------------------
app = FastAPI(
    title="Charlie MBTA Delay Prediction API",
    description="Serves the production delay model selected by the MLOps pipeline.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict whether a trip will be delayed based on minimal features.
    """
    logger.info(
        f"Received prediction request: direction_id={request.direction_id}, "
        f"stop_sequence={request.stop_sequence}"
    )
    result = predict_one(
        direction_id=request.direction_id,
        stop_sequence=request.stop_sequence,
    )
    return result


# ----------------------------------------------------
# CLI smoke test
# ----------------------------------------------------
if __name__ == "__main__":
    # Simple manual test without starting an API server
    logger.info("Running CLI smoke test for deploy_model.py")

    try:
        sample = predict_one(direction_id=0, stop_sequence=5)
        logger.info(
            f"Sample prediction -> "
            f"prob_delayed={sample.delayed_probability:.4f}, "
            f"label={sample.delayed_prediction}, "
            f"model={sample.model_name}"
        )
        print(
            f"[SMOKE TEST] prob_delayed={sample.delayed_probability:.4f}, "
            f"label={sample.delayed_prediction}, "
            f"model={sample.model_name}"
        )
    except Exception as e:
        logger.exception(f"Smoke test failed: {e}")
        raise