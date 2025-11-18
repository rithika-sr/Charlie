# Model_Development/ml_src/data_loader.py

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import yaml


class DataPaths:
    """
    Loads processed outputs from the data pipeline using the paths.yaml config.
    Works both locally AND inside GitHub Actions.
    """

    def __init__(self, cfg_path: str = "ml_configs/paths.yaml"):
        self.cfg_path = cfg_path

        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"❌ paths.yaml not found at {cfg_path}. "
                f"Ensure you run from project ROOT, not inside Model_Development/."
            )

        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Directories
        self.proc_dir = Path(self.cfg["data"]["processed_dir"])

        # Files inside processed/
        self.predictions = Path(self.cfg["data"]["predictions"])
        self.vehicles = Path(self.cfg["data"]["vehicles"])
        self.alerts = Path(self.cfg["data"]["alerts"])

        # Required files
        self.required = {
            "predictions": self.predictions,
            "vehicles": self.vehicles,
            "alerts": self.alerts,
        }

        # Validation rules
        self.min_rows = self.cfg.get("validation", {}).get("min_rows", {})

    # ----------------------------------------------------
    # Check files exist
    # ----------------------------------------------------
    def check_exists(self):
        missing = [name for name, path in self.required.items() if not path.exists()]

        if missing:
            raise FileNotFoundError(
                f"❌ Missing processed files: {missing}\n"
                f"Expected inside processed directory: {self.proc_dir}\n"
                f"➡️ Run the data pipeline first:\n"
                f"   cd Data_Pipeline && dvc pull\n"
            )

    # ----------------------------------------------------
    # Load all processed data as DataFrames
    # ----------------------------------------------------
    def load_all(self) -> dict[str, pd.DataFrame]:
        self.check_exists()

        return {
            "predictions": pd.read_csv(self.predictions),
            "vehicles": pd.read_csv(self.vehicles),
            "alerts": pd.read_csv(self.alerts),
        }


# ----------------------------------------------------------
# Quick sanity validation for debugging + assignment report
# ----------------------------------------------------------
def quick_sanity(dfs: dict[str, pd.DataFrame], min_rows: dict | None = None) -> dict:
    """
    Lightweight contract checks:
    - shape
    - non-empty
    - meets minimum row expectation
    - duplicate column names
    - null ratios
    """

    report = {}
    min_rows = min_rows or {}

    for name, df in dfs.items():
        stats = {}

        stats["shape"] = df.shape
        stats["non_empty"] = df.shape[0] > 0 and df.shape[1] > 0
        stats["min_rows_ok"] = df.shape[0] >= int(min_rows.get(name, 0))
        stats["duplicate_columns"] = df.columns.duplicated().any()
        stats["null_ratio_by_col"] = df.isna().mean().round(4).to_dict()

        report[name] = stats

    return report


# ----------------------------------------------------------
# Run manually: python -m Model_Development.ml_src.data_loader
# ----------------------------------------------------------
if __name__ == "__main__":
    paths = DataPaths()
    dfs = paths.load_all()
    rep = quick_sanity(dfs, paths.min_rows)

    print("\n===== SANITY CHECK REPORT =====")
    for name, stats in rep.items():
        print(f"\nDataset: {name}")
        print(f"  Shape: {stats['shape']}")
        print(f"  Non-empty: {stats['non_empty']}")
        print(f"  Minimum rows met: {stats['min_rows_ok']}")
        print(f"  Duplicate columns: {stats['duplicate_columns']}")
        print(f"  Null % by column: {stats['null_ratio_by_col']}")
    print("\n================================\n")