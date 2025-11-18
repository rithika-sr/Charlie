import os, json, yaml, requests, time
from datetime import datetime, UTC
from pathlib import Path
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def setup_logger(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

def fetch_and_save(url: str, out_dir: Path, prefix: str, retries=3):
    out_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        resp = requests.get(url, timeout=30)
        if resp.status_code == 429:
            wait = 60 * (attempt + 1)  # exponential backoff: 1min, 2min, 3min
            logging.warning(f"⚠️ Rate limit hit ({resp.status_code}). Waiting {wait}s before retry...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"{prefix}_{ts}.json"
        with open(out_file, "w") as f:
            json.dump(data, f, indent=2)
        return out_file, data
    raise RuntimeError(f"Failed to fetch {url} after {retries} retries")

def main():
    params = load_params()
    setup_logger(params["logging"]["level"])

    base_url = params["data"]["base_url"]
    raw_dir = Path(params["data"]["raw_dir"])
    endpoints = params["data"]["endpoints"]
    routes_file = Path(params["data"]["all_routes_file"])

    # Load routes dynamically if available; otherwise fall back to minimal set
    if routes_file.exists():
        with open(routes_file) as f:
            all_routes = yaml.safe_load(f)["routes"]
        logging.info(f"Loaded {len(all_routes)} routes from {routes_file}")
    else:
        all_routes = ["Red", "Orange", "Blue", "Green-B", "Green-C", "Green-D", "Green-E", "Mattapan"]
        logging.warning("No all_routes.yaml found — using default subway routes only.")

    # Fetch predictions PER ROUTE to avoid 400 errors and control payload size
    pred_dir = raw_dir / "predictions"
    for i, route in enumerate(all_routes, start=1):
        url = f"{base_url}{endpoints['predictions']}?route={route}"
        try:
            out_file, data = fetch_and_save(url, pred_dir, f"predictions_{route}")
            count = len(data.get("data", []))
            logging.info(f"✅ Saved predictions for {route} ({count} records) → {out_file}")
        except requests.HTTPError as e:
            logging.warning(f"Skipping route {route} due to HTTP {e.response.status_code}")
        except Exception as e:
            logging.exception(f"Route {route} failed: {e}")
        time.sleep(1)  # be polite to the API
        logging.info("Waiting 1 second before next route...")

    # Vehicles (single call)
    try:
        out_file, data = fetch_and_save(f"{base_url}{endpoints['vehicles']}", raw_dir / "vehicles", "vehicles")
        logging.info(f"✅ Saved vehicles data ({len(data.get('data', []))} records) → {out_file}")
    except Exception:
        logging.exception("Vehicles fetch failed.")

    # Alerts (single call)
    try:
        out_file, data = fetch_and_save(f"{base_url}{endpoints['alerts']}", raw_dir / "alerts", "alerts")
        logging.info(f"✅ Saved alerts data ({len(data.get('data', []))} records) → {out_file}")
    except Exception:
        logging.exception("Alerts fetch failed.")

    logging.info("All endpoints fetched.")

if __name__ == "__main__":
    main()