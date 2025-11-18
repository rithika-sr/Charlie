import pandas as pd
import json, logging
from ydata_profiling import ProfileReport
from pathlib import Path
import yaml
import great_expectations as ge


def setup_logger():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def validate_schema(df: pd.DataFrame, expected_cols: list[str]):
    """Validate column presence using Great Expectations logic."""
    actual = set(df.columns)
    expected = set(expected_cols)
    missing = list(expected - actual)
    extra = list(actual - expected)
    return {
        "missing_columns": missing,
        "extra_columns": extra,
        "status": "PASS" if not missing else "FAIL",
    }


def main():
    setup_logger()
    params = load_params()
    processed_dir = Path(params["data"]["processed_dir"])
    validation_dir = Path(params["data"]["validation_dir"])
    validation_dir.mkdir(parents=True, exist_ok=True)

    checks = {
        "predictions.csv": params["schema"]["predictions_expected"],
        "vehicles.csv": params["schema"]["vehicles_expected"],
        "alerts.csv": params["schema"]["alerts_expected"],
    }

    report = []
    for name, expected_cols in checks.items():
        file_path = processed_dir / name
        if not file_path.exists():
            logging.warning(f"{name} not found, skipping.")
            continue

        df = pd.read_csv(file_path, low_memory=False)

        # --- Profiling Report ---
        profile_path = validation_dir / f"{name}_profile.html"
        ProfileReport(df, title=f"{name} Profiling", minimal=True).to_file(profile_path)
        logging.info(f"✅ Profiling report saved → {profile_path}")

        # --- Schema Validation ---
        res = validate_schema(df, expected_cols)
        res["file"] = name
        report.append(res)
        logging.info(f"{name}: {res['status']}")

    out_path = validation_dir / "profiling_schema_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"📊 Combined profiling + schema report → {out_path}")


if __name__ == "__main__":
    main()