# Model_Development/ml_src/model_tuning.py

from __future__ import annotations
import os
import yaml
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

from Model_Development.ml_src.data_loader import DataPaths
from Model_Development.ml_src.utils.logging import get_logger
from Model_Development.ml_src.model_train import prepare_data

logger = get_logger("model_tuning")


# ============================================================
# HYPERPARAMETER TUNING FOR LOGISTIC REGRESSION
# ============================================================
def train_and_log():
    logger.info("🚀 Starting hyperparameter tuning...")

    # ------------------------------
    # Load processed dataset
    # ------------------------------
    paths = DataPaths("ml_configs/paths.yaml")
    try:
        df_pred = paths.load_all()["predictions"]
    except Exception as e:
        logger.error(f"❌ Cannot load processed data: {e}")
        return

    logger.info(f"📥 Loaded dataset: {df_pred.shape}")

    # ------------------------------
    # Prepare engineered features
    # ------------------------------
    _, X, y = prepare_data(df_pred)

    n_pos = (y == 1).sum()
    logger.info(f"🔍 Positive (delayed) samples = {n_pos}")

    # ------------------------------
    # SMOTE: Prevent crash when minority class is too small
    # ------------------------------
    if n_pos < 2:
        logger.warning("❌ Too few positive samples for SMOTE. Skipping tuning.")
        return

    smote = SMOTE(k_neighbors=1, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    logger.info(f"📈 After SMOTE: {X_res.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # ------------------------------
    # MLflow config
    # ------------------------------
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])  # file:./Model_Development/mlruns
    mlflow.set_experiment(cfg["experiment"]["name"])

    # ------------------------------
    # Hyperparameter grid
    # ------------------------------
    params = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"]
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=2000),
        params,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    logger.info(f"🏆 Best Parameters → {grid.best_params_}")

    # ------------------------------
    # Evaluate tuned model
    # ------------------------------
    y_pred = best_model.predict(X_val)
    y_prob = best_model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_prob),
    }

    logger.info(f"📊 Tuned Model Metrics: {metrics}")

    # ------------------------------
    # Log to MLflow
    # ------------------------------
    with mlflow.start_run(run_name="logreg_tuned"):
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, artifact_path="tuned_logreg")

        # ------------------------------
        # Save model to correct directory
        # ------------------------------
        model_dir = "Model_Development/models"
        os.makedirs(model_dir, exist_ok=True)

        model_path = f"{model_dir}/logreg_tuned.joblib"
        joblib.dump(best_model, model_path)

        logger.info(f"💾 Tuned model saved → {model_path}")

    logger.info("🎯 Hyperparameter tuning completed successfully!")


if __name__ == "__main__":
    train_and_log()