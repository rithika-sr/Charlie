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

def validate_csv(file_path: Path, expected_cols: list[str]):
    if not file_path.exists():
        return {"file": str(file_path), "status": "MISSING", "rows": 0}
    df = pd.read_csv(file_path, low_memory=False)
    actual = set(df.columns)
    expected = set(expected_cols)
    missing = list(expected - actual)
    extra = list(actual - expected)
    return {
        "file": str(file_path),
        "rows": int(len(df)),
        "missing_columns": missing,
        "extra_columns": extra,
        "status": "PASS" if not missing else "FAIL",
    }

def main():
    setup_logger()
    p = load_params()
    processed_dir = Path(p["data"]["processed_dir"])

    checks = {
        "predictions.csv": p["schema"]["predictions_expected"],
        "vehicles.csv": p["schema"]["vehicles_expected"],
        "alerts.csv": p["schema"]["alerts_expected"],
    }

    results = []
    for name, cols in checks.items():
        res = validate_csv(processed_dir / name, cols)
        results.append(res)
        logging.info(f"{name}: {res['status']} (rows={res.get('rows', 0)})")

    out_path = Path("Data_Pipeline/data/validation/schema_validation_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"✅ Validation complete → {out_path}")

if __name__ == "__main__":
    main()