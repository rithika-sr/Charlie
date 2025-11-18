import pandas as pd
import json
import yaml
import logging
from pathlib import Path
from ydata_profiling import ProfileReport
import great_expectations as ge

def setup_logger():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def safe_cast_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all datetime columns are stringified to avoid profiling errors."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or df[col].apply(lambda x: isinstance(x, pd.Timestamp) or isinstance(x, pd.Timestamp), axis=0).any():
            df[col] = df[col].astype(str)
    return df

def generate_profile(df, name, out_dir):
    df = safe_cast_dates(df)  # ✅ Prevents datetime errors
    profile = ProfileReport(df, title=f"{name} Profiling Report", explorative=True, minimal=True)
    out_path = out_dir / f"{name}_profiling.html"
    profile.to_file(out_path)
    logging.info(f"✅ Profiling report saved → {out_path}")

def validate_schema(df, expected_columns):
    ge_df = ge.from_pandas(df)
    results = {"missing_columns": [], "extra_columns": []}

    actual = set(df.columns)
    expected = set(expected_columns)

    results["missing_columns"] = list(expected - actual)
    results["extra_columns"] = list(actual - expected)
    results["status"] = "PASS" if not results["missing_columns"] else "FAIL"

    return results

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
            logging.warning(f"{name} not found.")
            continue
        df = pd.read_csv(file_path, low_memory=False)

        # Generate profile report
        try:
            generate_profile(df, name.replace(".csv", ""), validation_dir)
        except Exception as e:
            logging.warning(f"⚠️ Skipped profiling for {name} due to error: {e}")

        # Validate schema
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