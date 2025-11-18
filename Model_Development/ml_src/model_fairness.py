# Model_Development/ml_src/model_fairness.py

import os
import yaml
import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
)
from sklearn.metrics import accuracy_score, recall_score

from Model_Development.ml_src.data_loader import DataPaths
from Model_Development.ml_src.model_train import prepare_data
from Model_Development.ml_src.utils.logging import get_logger

logger = get_logger("model_fairness")


def evaluate_fairness():

    # ---------------------------------------------------------
    # Load MLflow Configuration
    # ---------------------------------------------------------
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])

    # ---------------------------------------------------------
    # Load processed predictions
    # ---------------------------------------------------------
    paths = DataPaths("ml_configs/paths.yaml")
    df_pred = paths.load_all()["predictions"]

    df_clean, X, y_true = prepare_data(df_pred)

    # ---------------------------------------------------------
    # Select best available model
    # ---------------------------------------------------------
    model_candidates = [
        "Model_Development/models/final_model.joblib",
        "Model_Development/models/model_lgbm.joblib",
        "Model_Development/models/logreg_tuned.joblib",
    ]

    model_path = next((m for m in model_candidates if os.path.exists(m)), None)
    if not model_path:
        raise FileNotFoundError("❌ No model found in Model_Development/models/")

    model = joblib.load(model_path)
    logger.info(f"📦 Loaded model → {model_path}")

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------
    y_pred = model.predict(X)

    # ---------------------------------------------------------
    # Sensitive Feature
    # ---------------------------------------------------------
    if "direction_id" not in X.columns:
        raise ValueError("❌ direction_id missing for fairness evaluation.")

    sensitive_feature = X["direction_id"]

    # ---------------------------------------------------------
    # Build MetricFrame
    # ---------------------------------------------------------
    metrics = {
        "accuracy": accuracy_score,
        "recall": recall_score,
        "selection_rate": selection_rate,
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature,
    )

    logger.info(f"\n📊 Fairness metrics by direction_id:\n{mf.by_group}")

    # ---------------------------------------------------------
    # Output Directory
    # ---------------------------------------------------------
    report_dir = Path("Model_Development/reports")
    report_dir.mkdir(exist_ok=True)

    fairness_plot_path = report_dir / "fairness_by_direction.png"
    fairness_csv_path = report_dir / "fairness_metrics.csv"

    # ---------------------------------------------------------
    # Plot Metrics
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 4))
    mf.by_group.plot(kind="bar")
    plt.title("Fairness Metrics by Direction ID")
    plt.ylabel("Metric Value")
    plt.tight_layout()
    plt.savefig(fairness_plot_path, dpi=300)
    plt.close()

    logger.info(f"📈 Fairness plot saved → {fairness_plot_path}")

    # Save CSV
    mf.by_group.to_csv(fairness_csv_path)
    logger.info(f"📄 Fairness metrics saved → {fairness_csv_path}")

    # ---------------------------------------------------------
    # Demographic Parity Difference
    # ---------------------------------------------------------
    dp_diff = demographic_parity_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature,
    )

    logger.info(f"🎯 Demographic Parity Difference = {dp_diff:.4f}")

    # ---------------------------------------------------------
    # Log to MLflow
    # ---------------------------------------------------------
    try:
        with mlflow.start_run(run_name="model_fairness"):
            mlflow.log_artifact(str(fairness_plot_path))
            mlflow.log_artifact(str(fairness_csv_path))
            mlflow.log_metric("demographic_parity_difference", dp_diff)

        logger.info("☁️ Logged fairness artifacts to MLflow")

    except Exception as e:
        logger.warning(f"⚠️ MLflow logging failed: {e}")

    logger.info("✅ Fairness Evaluation Completed!")


if __name__ == "__main__":
    evaluate_fairness()