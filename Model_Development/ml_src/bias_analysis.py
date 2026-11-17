# ml_src/bias_analysis.py
from __future__ import annotations
import os
import pandas as pd
import joblib
from pathlib import Path
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score
from Model_Development.ml_src.model_train import prepare_data
from Model_Development.ml_src.data_loader import DataPaths
from Model_Development.ml_src.utils.logging import get_logger

logger = get_logger("bias_analysis")


def main():
    Path("reports").mkdir(exist_ok=True)

    # -----------------------
    # Load model
    # -----------------------
    model_path_candidates = [
        "Model_Development/models/model_lgbm.joblib",
        "Model_Development/models/best_model.joblib",
        "Model_Development/models/baseline_logreg.joblib",
    ]

    model_path = next((p for p in model_path_candidates if os.path.exists(p)), None)
    if not model_path:
        logger.error("❌ No trained model found for bias analysis. Run model_train.py first.")
        raise SystemExit(1)

    model = joblib.load(model_path)
    logger.info(f"📦 Loaded model from {model_path}")

    # -----------------------
    # Load processed data
    # -----------------------
    paths = DataPaths("ml_configs/paths.yaml")
    try:
        df_pred = paths.load_all()["predictions"]
    except Exception as e:
        logger.error(f"❌ Cannot load processed data for bias analysis: {e}")
        raise SystemExit(1)

    # -----------------------
    # Prepare data (FIXED)
    # -----------------------
    df_eng, X, y = prepare_data(df_pred)

    if "direction_id" not in X.columns:
        logger.error("❌ direction_id missing from the features; bias analysis can't run.")
        raise SystemExit(1)

    sensitive = X["direction_id"]
    y_pred = model.predict(X)

    # -----------------------
    # MetricFrame
    # -----------------------
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score},
        y_true=y,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    results = {
        "overall_accuracy": accuracy_score(y, y_pred),
        "group_accuracy": mf.by_group.to_dict(),
        "difference": mf.difference(),
    }

    pd.DataFrame.from_dict(results).to_csv("reports/bias_report.csv", index=False)
    logger.info("Bias report saved → reports/bias_report.csv")


if __name__ == "__main__":
    main()