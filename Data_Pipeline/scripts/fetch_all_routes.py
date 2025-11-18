import sys
from pathlib import Path

# ✅ Add this so DVC and Airflow can import modules properly
sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import requests
from Data_Pipeline.scripts.utils import ensure_dir
import yaml  

MBTA_ROUTES_URL = "https://api-v3.mbta.com/routes"
DEFAULT_OUTDIR = Path("Data_Pipeline/data")

def get_routes(session: Optional[requests.Session] = None, max_retries: int = 3) -> List[Dict[str, Any]]:
    """Fetch routes from MBTA with simple 429 backoff."""
    sess = session or requests.Session()
    for attempt in range(1, max_retries + 1):
        resp = sess.get(MBTA_ROUTES_URL, timeout=30)
        if resp.status_code == 429 and attempt < max_retries:
            time.sleep(60)  # backoff on rate limit
            continue
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    # If we got here, final call will raise
    resp = sess.get(MBTA_ROUTES_URL, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])

def save_routes(routes: List[Dict[str, Any]], outdir: Path = DEFAULT_OUTDIR) -> Path:
    ensure_dir(outdir)
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    outpath = outdir / f"routes_{ts}.json"
    with outpath.open("w") as f:
        json.dump({"data": routes}, f, indent=2)
    return outpath

def main() -> None:
    routes = get_routes()
    out = save_routes(routes, DEFAULT_OUTDIR)
    print(f"✅ Saved routes → {out}")

    # ✅ Save all_routes.json for next pipeline stages
    all_routes_path = DEFAULT_OUTDIR / "all_routes.json"
    with open(all_routes_path, "w") as f:
        json.dump({"routes": [r.get("id") for r in routes]}, f, indent=2)
    print(f"✅ Also saved route summary → {all_routes_path}")

    # ✅ Save all_routes.yaml (required by DVC)
    all_routes_yaml = DEFAULT_OUTDIR / "all_routes.yaml"
    with open(all_routes_yaml, "w") as f:
        yaml.safe_dump({"routes": [r.get("id") for r in routes]}, f)
    print(f"✅ Also saved YAML summary → {all_routes_yaml}")

if __name__ == "__main__":
    main()