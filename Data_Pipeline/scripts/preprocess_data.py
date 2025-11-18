import json
import pandas as pd
import yaml
import logging
import sys
from pathlib import Path
from typing import Iterable

# ✅ Add project root to PYTHONPATH before any local imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Data_Pipeline.scripts.utils import ensure_dir

# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------

def load_params():
    """Load configuration parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def setup_logger():
    """Configure basic logging"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def flatten_data(json_obj):
    """Flatten nested JSON from MBTA API responses"""
    if "data" not in json_obj or not json_obj["data"]:
        return pd.DataFrame()
    return pd.json_normalize(json_obj["data"])

def latest_files(folder: Path):
    """Return all .json files sorted by name (latest last)"""
    files = sorted(folder.glob("*.json"))
    return files

# ----------------------------------------------------------------------
# Reusable Function for Testing
# ----------------------------------------------------------------------

def preprocess_predictions_jsons(json_files: Iterable[Path], out_csv: Path) -> Path:
    """
    Read raw prediction JSONs (list of MBTA /predictions responses),
    normalize into a clean table, and save as CSV.
    Expected top-level shape per file: {"data": [...]}.
    """
    rows = []
    for fp in json_files:
        with open(fp, "r") as f:
            obj = json.load(f)
        for rec in obj.get("data", []):
            attrs = rec.get("attributes", {}) or {}
            relationships = rec.get("relationships", {}) or {}
            rid = relationships.get("route", {}).get("data", {}).get("id")
            rows.append({
                "route_id": rid,
                "arrival_time": attrs.get("arrival_time"),
                "departure_time": attrs.get("departure_time"),
                "direction_id": attrs.get("direction_id"),
                "status": attrs.get("status"),
                "stop_sequence": attrs.get("stop_sequence"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["direction_id"] = pd.to_numeric(df["direction_id"], errors="coerce").astype("Int64")
        df["stop_sequence"] = pd.to_numeric(df["stop_sequence"], errors="coerce").astype("Int64")
        df = df.drop_duplicates().reset_index(drop=True)

    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    return out_csv

# ----------------------------------------------------------------------
# Main Preprocessing Function (used in DAG)
# ----------------------------------------------------------------------

def preprocess():
    """Main entry point for Airflow DAG — merges, cleans, and exports all MBTA data"""
    params = load_params()
    setup_logger()

    raw_dir = Path(params["data"]["raw_dir"])
    processed_dir = Path(params["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ----- PREDICTIONS -----
    pred_dir = raw_dir / "predictions"
    pred_files = latest_files(pred_dir)
    if pred_files:
        out_csv = processed_dir / "predictions.csv"
        preprocess_predictions_jsons(pred_files, out_csv)
        logging.info(f"✅ Saved processed predictions → {out_csv}")
    else:
        logging.warning("No prediction frames found to process.")

    # ----- VEHICLES -----
    veh_dir = raw_dir / "vehicles"
    veh_files = latest_files(veh_dir)
    if veh_files:
        with open(veh_files[-1]) as jf:
            vobj = json.load(jf)
        vdf = flatten_data(vobj)
        if not vdf.empty:
            vdf.to_csv(processed_dir / "vehicles.csv", index=False)
            logging.info(f"✅ Saved processed vehicles → {processed_dir/'vehicles.csv'} (rows={len(vdf)})")
        else:
            logging.warning("Vehicles JSON was empty.")
    else:
        logging.warning("No vehicles files found.")

    # ----- ALERTS -----
    al_dir = raw_dir / "alerts"
    al_files = latest_files(al_dir)
    if al_files:
        with open(al_files[-1]) as jf:
            aobj = json.load(jf)
        adf = flatten_data(aobj)
        if not adf.empty:
            adf.to_csv(processed_dir / "alerts.csv", index=False)
            logging.info(f"✅ Saved processed alerts → {processed_dir/'alerts.csv'} (rows={len(adf)})")
        else:
            logging.warning("Alerts JSON was empty.")
    else:
        logging.warning("No alerts files found.")

# ----------------------------------------------------------------------
# Script Entrypoint
# ----------------------------------------------------------------------

if __name__ == "__main__":
    preprocess()