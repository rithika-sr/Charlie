# Model_Development/ml_src/register_model.py

import os
import joblib
import mlflow
import mlflow.sklearn

from Model_Development.ml_src.utils.logging import get_logger

logger = get_logger("register_model")

# Correct folder paths (consistent across all pipeline scripts)
MODEL_DIR = "Model_Development/models"
BEST_MODEL_FILE = "final_model.joblib"
MODEL_NAME = "Charlie_MBTA_Model"   # MLflow Registry Name


def load_best_model_path():
    """Return path to the final model inside Model_Development/models."""
    final_path = os.path.join(MODEL_DIR, BEST_MODEL_FILE)

    if not os.path.exists(final_path):
        raise FileNotFoundError(f"❌ No final model found at {final_path}")

    logger.info(f"✅ Found final model → {final_path}")
    return final_path


def register_model_with_mlflow(model_path):
    """Register the model inside the MLflow Model Registry."""

    # MLflow registry stored inside: Model_Development/mlruns/
    mlflow.set_tracking_uri("file:./Model_Development/mlruns")
    mlflow.set_experiment("MBTA-Model-Registry")

    with mlflow.start_run(run_name="register_best_model"):

        # Load model
        model = joblib.load(model_path)
        logger.info("📦 Logging model to MLflow registry…")

        # Log & register model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        logger.info("✅ Model logged to MLflow registry")

        # Retrieve the latest model version
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging", "Production"])

        latest_version = versions[-1].version
        logger.info(f"📌 Latest registered version = {latest_version}")

        # Promote to PRODUCTION
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )

        logger.info(f"🚀 Model promoted to PRODUCTION (version {latest_version})")
        return latest_version


def main():
    try:
        model_path = load_best_model_path()
        version = register_model_with_mlflow(model_path)
        logger.info(f"🎉 Model successfully registered at version {version}")

    except Exception as e:
        logger.error(f"❌ Model registration failed: {str(e)}")


if __name__ == "__main__":
    main()