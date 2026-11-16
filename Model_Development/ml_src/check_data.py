# Model_Development/ml_src/check_data.py
"""
Gate for ML validation: the processed training data must be present, large enough,
and exactly the version recorded in dvc.lock. Exits non-zero (and prints a GitHub
Actions ::error::) otherwise, so missing data can never look like a passing run.

Run from the project root:  python -m Model_Development.ml_src.check_data
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from Model_Development.ml_src.data_loader import DataPaths

LOCK_FILE = Path("dvc.lock")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locked_hashes(lock_file: Path = LOCK_FILE) -> dict[str, str]:
    """{path: md5} for every stage output recorded in dvc.lock."""
    lock = yaml.safe_load(lock_file.read_text())
    out = {}
    for stage in (lock.get("stages") or {}).values():
        for o in stage.get("outs") or []:
            if "md5" in o:
                out[o["path"]] = o["md5"]
    return out


def check(paths: DataPaths, lock: dict[str, str]) -> tuple[list[str], list[str]]:
    """(errors, report lines)."""
    errors, report = [], []
    for name, path in paths.required.items():
        if not path.exists():
            errors.append(f"{path} is missing")
            continue
        rows = len(pd.read_csv(path))
        need = int(paths.min_rows.get(name, 0))
        expected = lock.get(str(path))
        actual = md5(path)
        report.append(f"| `{path}` | {rows:,} | {actual[:12]} | {'matches dvc.lock' if actual == expected else 'NOT in dvc.lock' if not expected else 'DIFFERS from dvc.lock'} |")
        if rows < need:
            errors.append(f"{path} has {rows} rows; at least {need} are required")
        if expected and actual != expected:
            errors.append(f"{path} (md5 {actual}) is not the version pinned in dvc.lock (md5 {expected})")
        if not expected:
            errors.append(f"{path} has no pinned version in dvc.lock")
    return errors, report


def main() -> int:
    paths = DataPaths("ml_configs/paths.yaml")
    errors, report = check(paths, locked_hashes())
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    lines = ["### Training data", "", "| File | Rows | md5 | Version |", "|---|---|---|---|", *report, ""]
    if errors:
        lines += ["**ML validation cannot run:**", *[f"- {e}" for e in errors], "",
                  "See `Model_Development/readme.md` → *Training data in CI* for how the data is provided."]
    if summary:
        with open(summary, "a") as f:
            f.write("\n".join(lines) + "\n")
    for e in errors:
        print(f"::error title=Training data unavailable::{e}")
    if errors:
        print("❌ Required training data is missing or not the pinned version; ML validation did not run.")
        return 1
    print("✅ Training data present and matches dvc.lock.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
