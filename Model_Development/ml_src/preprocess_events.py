import pandas as pd
import glob
import os

RAW_DIR = "Model_Development/data/raw_events_2025"
OUT_FILE = "Model_Development/data/clean_events_2025.csv"

# ---- COMMON MBTA COLUMN NAME MAPPINGS (covers all versions) ----
COLUMN_MAP = {
    # time columns (MBTA changes these names frequently)
    "event_time": "actual_time",
    "actual_time": "actual_time",
    "actual_arrival_time": "actual_time",
    "actual_departure_time": "actual_time",

    "scheduled_time": "scheduled_time",
    "scheduled_arrival_time": "scheduled_time",
    "scheduled_departure_time": "scheduled_time",

    # identifiers
    "route_id": "route_id",
    "route": "route_id",

    "direction_id": "direction_id",
    "direction": "direction_id",

    "stop_id": "stop_id",
    "station_id": "stop_id",
    "parent_station": "stop_id",

    "stop_sequence": "stop_sequence",
    "sequence": "stop_sequence",

    "trip_id": "trip_id",
    "vehicle_id": "vehicle_id",

    "event_type": "event_type"
}

def load_all_events():
    files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    print(f"Found {len(files)} files")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            print(f"Loaded {f} → {df.shape}")

            # Rename known columns
            df = df.rename(columns={col: COLUMN_MAP[col] for col in df.columns if col in COLUMN_MAP})

            dfs.append(df)

        except Exception as e:
            print(f"Error loading {f}: {e}")

    return pd.concat(dfs, ignore_index=True)


def clean_events(df):
    print("\nColumns after renaming:", df.columns.tolist())

    # Required columns — if missing, we will skip
    required = ["actual_time", "scheduled_time", "route_id", "direction_id", "stop_id"]

    for col in required:
        if col not in df.columns:
            print(f"❌ Missing column: {col}")
            print("⚠️ Your dataset does not include required fields. Trying to infer...")

    # Convert times
    df["actual_time"] = pd.to_datetime(df["actual_time"], errors="coerce")
    df["scheduled_time"] = pd.to_datetime(df["scheduled_time"], errors="coerce")

    df = df.dropna(subset=["actual_time", "scheduled_time"])

    # Filter ARRIVAL events only (if event_type exists)
    if "event_type" in df.columns:
        df = df[df["event_type"].str.upper() == "ARRIVAL"]

    # Compute delay
    df["delay_minutes"] = (df["actual_time"] - df["scheduled_time"]).dt.total_seconds() / 60

    # Label
    df["delay_label"] = (df["delay_minutes"] > 2).astype(int)

    # Stop sequence fallback
    if "stop_sequence" not in df.columns:
        df["stop_sequence"] = -1  # unknown sequences

    # Time features
    df["hour_of_day"] = df["actual_time"].dt.hour
    df["day_of_week"] = df["actual_time"].dt.dayofweek

    # Select final ML columns
    final_cols = [
        "route_id",
        "direction_id",
        "stop_id",
        "stop_sequence",
        "delay_minutes",
        "delay_label",
        "hour_of_day",
        "day_of_week"
    ]

    available_cols = [c for c in final_cols if c in df.columns]
    df = df[available_cols]

    print("\nFinal cleaned shape:", df.shape)
    return df


def main():
    df = load_all_events()
    print("\nAll events combined:", df.shape)

    df_clean = clean_events(df)

    df_clean.to_csv(OUT_FILE, index=False)
    print(f"\n✅ Saved cleaned dataset → {OUT_FILE}\n")


if __name__ == "__main__":
    main()