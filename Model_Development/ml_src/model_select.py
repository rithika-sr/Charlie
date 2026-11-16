# Model_Development/ml_src/model_select.py

import os
import json
import joblib
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from Model_Development.ml_src.data_loader import DataPaths
from Model_Development.ml_src.model_train import prepare_data
from Model_Development.ml_src.utils.logging import get_logger
from Model_Development.ml_src.utils.mlflow_models import log_sklearn_model

logger = get_logger("model_select")


def load_model_safe(path):
    """Load model safely; return None if missing."""
    if not os.path.exists(path):
        logger.warning(f"⚠️ Model not found: {path}")
        return None
    return joblib.load(path)


def evaluate_model(model, X, y):
    """Evaluate the model and return metrics."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
    }


def main():
    logger.info("🚀 Starting Model Comparison & Final Model Selection...")

    # ----------------------------------------------------------
    # 1️⃣ Load processed data
    # ----------------------------------------------------------
    paths = DataPaths("ml_configs/paths.yaml")
    df_pred = paths.load_all()["predictions"]

    df_clean, X, y = prepare_data(df_pred)

    # ----------------------------------------------------------
    # 2️⃣ Load available models from Model_Development/models/
    # ----------------------------------------------------------
    model_dir = Path("Model_Development/models")
    if not model_dir.exists():
        raise FileNotFoundError("❌ Missing Model_Development/models directory.")

    model_candidates = {
        "baseline_lgbm": model_dir / "model_lgbm.joblib",
        "logreg_tuned": model_dir / "logreg_tuned.joblib",
        "final_model_existing": model_dir / "final_model.joblib",
    }

    loaded_models = {}
    for name, path in model_candidates.items():
        model = load_model_safe(str(path))
        if model is not None:
            loaded_models[name] = model

    if not loaded_models:
        raise RuntimeError("❌ No models found for comparison.")

    logger.info(f"📦 Models available: {list(loaded_models.keys())}")

    # ----------------------------------------------------------
    # 3️⃣ Evaluate models
    # ----------------------------------------------------------
    results = {}
    for name, model in loaded_models.items():
        metrics = evaluate_model(model, X, y)
        results[name] = metrics
        logger.info(f"📊 {name} → {metrics}")

    # ----------------------------------------------------------
    # 4️⃣ Select best model by ROC-AUC
    # ----------------------------------------------------------
    best_model_name = max(results, key=lambda m: results[m]["roc_auc"])
    best_model = loaded_models[best_model_name]

    logger.info(f"🏆 Best model selected: {best_model_name}")

    # ----------------------------------------------------------
    # 5️⃣ Save FINAL model
    # ----------------------------------------------------------
    final_model_path = model_dir / "final_model.joblib"
    joblib.dump(best_model, final_model_path)
    logger.info(f"💾 Saved final model → {final_model_path}")

    # ----------------------------------------------------------
    # 6️⃣ Save JSON report
    # ----------------------------------------------------------
    report_dir = Path("Model_Development/reports")
    report_dir.mkdir(exist_ok=True)

    comparison_json = report_dir / "model_comparison.json"
    with open(comparison_json, "w") as f:
        json.dump(
            {"results": results, "best_model": best_model_name},
            f,
            indent=4
        )

    logger.info(f"📄 Saved comparison report → {comparison_json}")

    # ----------------------------------------------------------
    # 7️⃣ Save comparison plot
    # ----------------------------------------------------------
    plt.figure(figsize=(8, 5))
    model_names = list(results.keys())
    auc_scores = [results[m]["roc_auc"] for m in model_names]

    plt.bar(model_names, auc_scores, color="skyblue")
    plt.title("Model Comparison (ROC-AUC)")
    plt.ylabel("ROC-AUC")
    plt.xticks(rotation=20)
    plt.tight_layout()

    comparison_plot = report_dir / "model_comparison.png"
    plt.savefig(comparison_plot, dpi=300)
    plt.close()

    logger.info(f"📊 Saved comparison plot → {comparison_plot}")

    # ----------------------------------------------------------
    # 8️⃣ MLflow logging
    # ----------------------------------------------------------
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])  # file:./Model_Development/mlruns
    mlflow.set_experiment(cfg["experiment"]["name"])

    try:
        with mlflow.start_run(run_name="model_selection"):
            # metrics
            for name, m in results.items():
                mlflow.log_metric(f"{name}_auc", m["roc_auc"])
                mlflow.log_metric(f"{name}_accuracy", m["accuracy"])
                mlflow.log_metric(f"{name}_f1", m["f1"])

            mlflow.log_param("selected_model", best_model_name)

            mlflow.log_artifact(str(comparison_json))
            mlflow.log_artifact(str(comparison_plot))

            log_sklearn_model(best_model, "final_model")

        logger.info("📌 Model selection logged to MLflow.")

    except Exception as e:
        logger.error(f"❌ MLflow logging failed: {e}")
        raise SystemExit(1)

    logger.info("🎯 Model Selection Completed Successfully!")


if __name__ == "__main__":
    main()