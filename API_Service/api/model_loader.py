# api/model_loader.py
"""
ML Model Loader for Delay Prediction

Loads the trained delay prediction model and provides inference.
This is optional - the system works without the ML model.
"""

import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger("model_loader")

FEATURE_COLUMNS = ["direction_id", "stop_sequence"]

DEFAULT_MODEL_PATHS = [
    "models/final_model.joblib",
    "models/model_lgbm.joblib",
    "../models/final_model.joblib",
]


class DelayModel:
    """Wrapper for the trained delay prediction model."""
    
    def __init__(self) -> None:
        self.model = None
        self.model_path: Optional[str] = None
        self._load_model()
    
    def _find_model_path(self) -> Optional[str]:
        env_path = os.getenv("MODEL_PATH")
        if env_path and os.path.exists(env_path):
            return env_path
        
        for path in DEFAULT_MODEL_PATHS:
            if os.path.exists(path):
                return os.path.abspath(path)
        
        return None
    
    def _load_model(self) -> None:
        model_path = self._find_model_path()
        
        if not model_path:
            logger.warning("No delay prediction model found. Delay predictions disabled.")
            return
        
        try:
            import joblib
            self.model = joblib.load(model_path)
            self.model_path = model_path
            logger.info(f"Loaded delay model from: {model_path}")
        except ImportError as e:
            logger.warning(f"ML dependencies not installed: {e}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def is_available(self) -> bool:
        return self.model is not None
    
    def predict(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make a delay prediction."""
        if not self.is_available():
            return None
        
        try:
            import pandas as pd
            
            for col in FEATURE_COLUMNS:
                if col not in features:
                    return None
            
            df = pd.DataFrame({
                "direction_id": [int(features["direction_id"])],
                "stop_sequence": [int(features["stop_sequence"])],
            })
            
            if hasattr(self.model, "predict_proba"):
                proba = float(self.model.predict_proba(df)[:, 1][0])
            else:
                pred = int(self.model.predict(df)[0])
                proba = float(pred)
            
            label = int(proba >= 0.5)
            
            return {
                "probability_delay": round(proba, 4),
                "predicted_label": label,
                "delay_likely": label == 1,
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None


_model_instance: Optional[DelayModel] = None


def get_model() -> DelayModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = DelayModel()
    return _model_instance