import pandas as pd
import yaml
import logging
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def detect_pred_anomalies(df: pd.DataFrame) -> dict:
    out = {}
    if "attributes.arrival_time" in df.columns:
        out["missing_arrival_time"] = int(df["attributes.arrival_time"].isna().sum())
    return out

def detect_vehicle_anomalies(df: pd.DataFrame) -> dict:
    out = {}
    if "attributes.latitude" in df.columns:
        out["invalid_latitude_rows"] = int(((df["attributes.latitude"] < -90) | (df["attributes.latitude"] > 90)).sum())
    if "attributes.longitude" in df.columns:
        out["invalid_longitude_rows"] = int(((df["attributes.longitude"] < -180) | (df["attributes.longitude"] > 180)).sum())
    return out

def main():
    setup_logger()
    p = load_params()
    processed_dir = Path(p["data"]["processed_dir"])
    thresholds = p["anomaly_thresholds"]

    report = []

    # Predictions
    pred_path = processed_dir / "predictions.csv"
    if pred_path.exists():
        pdf = pd.read_csv(pred_path, low_memory=False)
        pa = detect_pred_anomalies(pdf)
        pa.update({"file": "predictions.csv"})
        report.append(pa)

    # Vehicles
    veh_path = processed_dir / "vehicles.csv"
    if veh_path.exists():
        vdf = pd.read_csv(veh_path, low_memory=False)
        va = detect_vehicle_anomalies(vdf)
        va.update({"file": "vehicles.csv"})
        report.append(va)

    # Alerts
    al_path = processed_dir / "alerts.csv"
    if al_path.exists():
        adf = pd.read_csv(al_path, low_memory=False)
        ra = {"file": "alerts.csv"}
        report.append(ra)

    # Save anomaly report
    out_path = Path("Data_Pipeline/data/validation/anomaly_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"✅ Saved anomaly report → {out_path}")

    # Threshold-based warning + trigger alert
    for item in report:
        if item.get("file") == "predictions.csv":
            if item.get("missing_arrival_time", 0) > thresholds["missing_arrival_time"]:
                logging.warning("⚠️ Missing arrival time exceeds threshold.")
                raise Exception("Test Airflow email alert: Too many missing arrival times detected.")

        if item.get("file") == "vehicles.csv":
            if item.get("invalid_latitude_rows", 0) > thresholds["invalid_latitude"]:
                logging.warning("⚠️ Invalid latitude rows exceed threshold.")
                raise Exception("Test Airflow email alert: Invalid latitude rows exceed threshold.")

            if item.get("invalid_longitude_rows", 0) > thresholds["invalid_longitude"]:
                logging.warning("⚠️ Invalid longitude rows exceed threshold.")
                raise Exception("Test Airflow email alert: Invalid longitude rows exceed threshold.")

if __name__ == "__main__":
    main()