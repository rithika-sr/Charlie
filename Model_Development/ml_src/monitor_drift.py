# Model_Development/ml_src/monitor_drift.py

import os
import json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import mlflow

from Model_Development.ml_src.utils.logging import get_logger
from Model_Development.ml_src.data_loader import DataPaths

logger = get_logger("monitor_drift")

REFERENCE_FILE = "Model_Development/models/reference_stats.json"
DRIFT_REPORT_JSON = "Model_Development/reports/drift_report.json"
DRIFT_REPORT_HTML = "Model_Development/reports/drift_report.html"


# ----------------------------------------------------
# Population Stability Index (PSI)
# ----------------------------------------------------
def calculate_psi(expected, actual, buckets=10):
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    expected_percents, _ = np.histogram(expected, bins=buckets)
    actual_percents, _ = np.histogram(actual, bins=buckets)

    expected_ratios = expected_percents / len(expected)
    actual_ratios = actual_percents / len(actual)

    # Convert to python float
    psi = float(np.sum((expected_ratios - actual_ratios) *
                       np.log((expected_ratios + 1e-6) /
                              (actual_ratios + 1e-6))))
    return psi


# ----------------------------------------------------
# Safe loader for reference data
# ----------------------------------------------------
def load_reference_array(ref_entry):
    if isinstance(ref_entry, dict) and "values" in ref_entry:
        return np.array(ref_entry["values"], dtype=float)

    if isinstance(ref_entry, list):
        return np.array(ref_entry, dtype=float)

    raise ValueError("Unexpected structure in reference_stats.json")


# ----------------------------------------------------
# Convert all numpy types → python types (REQUIRED)
# ----------------------------------------------------
def pythonize(obj):
    """Convert numpy values in dict into python-native serializable types."""
    if isinstance(obj, dict):
        return {k: pythonize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [pythonize(v) for v in obj]
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ----------------------------------------------------
# Drift Monitoring
# ----------------------------------------------------
def run_drift_monitoring():
    logger.info("🔍 Starting Drift Monitoring...")

    if not os.path.exists(REFERENCE_FILE):
        logger.error("❌ Reference stats missing. Run model_train.py first.")
        return

    # Load reference stats
    with open(REFERENCE_FILE, "r") as f:
        reference_stats = json.load(f)

    # Load processed prediction dataset
    paths = DataPaths("ml_configs/paths.yaml")
    df_pred = paths.load_all()["predictions"]

    # Prepare target
    df_pred["arrival_time"] = pd.to_datetime(df_pred["arrival_time"], errors="coerce")
    df_pred["departure_time"] = pd.to_datetime(df_pred["departure_time"], errors="coerce")
    df_pred["delay_minutes"] = (
        df_pred["departure_time"] - df_pred["arrival_time"]
    ).dt.total_seconds() / 60
    df_pred["delayed"] = (df_pred["delay_minutes"] > 5).astype(int)

    report = {"feature_drift": {}, "target_drift": {}, "psi_scores": {}}

    # -----------------------------
    # Feature drift
    # -----------------------------
    for feature in ["direction_id", "stop_sequence"]:
        if feature not in reference_stats:
            continue

        ref = load_reference_array(reference_stats[feature])
        cur = df_pred[feature].dropna().astype(float).values

        ks_stat, ks_p = ks_2samp(ref, cur)

        report["feature_drift"][feature] = {
            "ks_stat": float(ks_stat),
            "p_value": float(ks_p),
            "drift_detected": bool(ks_p < 0.05)
        }

        report["psi_scores"][feature] = calculate_psi(ref, cur)

    # -----------------------------
    # Target drift
    # -----------------------------
    if "delayed" in reference_stats:
        ref_target = load_reference_array(reference_stats["delayed"])
        cur_target = df_pred["delayed"].astype(float).values

        ks_stat, ks_p = ks_2samp(ref_target, cur_target)

        report["target_drift"] = {
            "ks_stat": float(ks_stat),
            "p_value": float(ks_p),
            "drift_detected": bool(ks_p < 0.05)
        }

    # -----------------------------
    # Convert numpy → python types
    # -----------------------------
    report_clean = pythonize(report)

    # -----------------------------
    # Save reports
    # -----------------------------
    os.makedirs("Model_Development/reports", exist_ok=True)

    with open(DRIFT_REPORT_JSON, "w") as f:
        json.dump(report_clean, f, indent=4)

    logger.info(f"📄 Drift report saved: {DRIFT_REPORT_JSON}")

    html = "<h1>MBTA Drift Monitoring Report</h1><pre>" + json.dumps(report_clean, indent=4) + "</pre>"
    with open(DRIFT_REPORT_HTML, "w") as f:
        f.write(html)

    logger.info(f"📊 Drift HTML written: {DRIFT_REPORT_HTML}")

    # -----------------------------
    # MLflow Logging
    # -----------------------------
    mlflow.set_tracking_uri("file:./Model_Development/mlruns")
    mlflow.set_experiment("MBTA-Model-Drift")

    with mlflow.start_run(run_name="drift_monitoring"):
        mlflow.log_artifact(DRIFT_REPORT_JSON)
        mlflow.log_artifact(DRIFT_REPORT_HTML)

        # PSI scores
        for feat, psi in report_clean["psi_scores"].items():
            mlflow.log_metric(f"psi_{feat}", psi)

    logger.info("🎯 Drift Monitoring Completed Successfully!")


if __name__ == "__main__":
    run_drift_monitoring()