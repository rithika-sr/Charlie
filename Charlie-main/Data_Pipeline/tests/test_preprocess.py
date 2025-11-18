import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import os
import json
import pandas as pd
from pathlib import Path
from Data_Pipeline.scripts import preprocess_data as prep

# --- Setup a temporary directory for test files ---
def setup_temp_files(tmp_path):
    data = {
        "data": [
            {
                "attributes": {
                    "arrival_time": "2025-01-01T10:00:00Z",
                    "departure_time": "2025-01-01T10:05:00Z",
                    "direction_id": 0,
                    "status": "On Time",
                    "stop_sequence": 3,
                },
                "relationships": {"route": {"data": {"id": "Red"}}},
            }
        ]
    }
    json_path = tmp_path / "test_predictions.json"
    with open(json_path, "w") as f:
        json.dump(data, f)
    return [json_path]

# --- Test flattening and CSV creation ---
def test_preprocess_predictions_jsons(tmp_path):
    files = setup_temp_files(tmp_path)
    out_csv = tmp_path / "out.csv"

    prep.preprocess_predictions_jsons(files, out_csv)

    assert out_csv.exists(), "Output CSV not created"
    df = pd.read_csv(out_csv)
    assert "route_id" in df.columns
    assert not df.empty

# --- Test flatten_data utility ---
def test_flatten_data():
    sample = {"data": [{"id": 1, "attributes": {"a": 10}}]}
    df = prep.flatten_data(sample)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

# --- Test latest_files ordering ---
def test_latest_files(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    files = prep.latest_files(tmp_path)
    assert files[0].name == "a.json"
    assert files[-1].name == "b.json"