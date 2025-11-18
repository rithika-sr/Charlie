import pandas as pd
import numpy as np
import logging
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score, f1_score

# -------------------------------------------------------
# Setup logging
# -------------------------------------------------------
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )


# -------------------------------------------------------
# Derive contextual features for slicing
# -------------------------------------------------------
def derive_time_features(df, timestamp_col="timestamp"):
    if timestamp_col not in df.columns:
        logging.warning(f"⚠️ '{timestamp_col}' column missing — skipping temporal features.")
        return df

    df["timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["time_of_day"] = pd.cut(
        df["hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["Night", "Morning", "Afternoon", "Evening"],
        include_lowest=True,
        right=False,
    )
    logging.info("✅ Derived time_of_day from timestamp column.")
    return df


# -------------------------------------------------------
# Bias detection with auto-slicing
# -------------------------------------------------------
def detect_bias(df):
    setup_logger()
    logging.info("🚦 Starting bias detection...")

    # Create simulated outcome if no 'actual' column
    if "actual" not in df.columns:
        logging.warning("⚠️ 'actual' column not found — simulating binary target variable 'on_time'.")
        np.random.seed(42)
        df["on_time"] = np.random.choice([0, 1], size=len(df))
    else:
        df["on_time"] = df["actual"]

    # Simulate predicted outcomes
    if "predicted" not in df.columns:
        df["predicted"] = df["on_time"] ^ np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])

    # Auto-detect candidate features for bias slicing
    candidate_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if "time_of_day" in df.columns and "time_of_day" not in candidate_features:
        candidate_features.append("time_of_day")

    if not candidate_features:
        logging.warning("⚠️ No categorical features found for bias slicing.")
        return None

    logging.info(f"📊 Auto-detected slicing features: {candidate_features[:5]}")
    results = []

    for feature in candidate_features[:5]:
        try:
            mf = MetricFrame(
                metrics={
                    "accuracy": accuracy_score,
                    "f1": f1_score,
                    "selection_rate": selection_rate,
                },
                y_true=df["on_time"],
                y_pred=df["predicted"],
                sensitive_features=df[feature],
            )
            group_metrics = mf.by_group.round(3)
            group_metrics["feature"] = feature
            results.append(group_metrics)
            logging.info(f"✅ Bias metrics computed for '{feature}'.")
        except Exception as e:
            logging.warning(f"⚠️ Error computing metrics for {feature}: {e}")

    if not results:
        logging.warning("⚠️ No bias results generated.")
        return None

    combined = pd.concat(results)
    output_path = "Data_Pipeline/data/validation/bias_report.csv"
    combined.to_csv(output_path)
    logging.info(f"📊 Bias report saved → {output_path}")
    return combined


# -------------------------------------------------------
# Main execution
# -------------------------------------------------------
if __name__ == "__main__":
    setup_logger()
    df = pd.read_csv("Data_Pipeline/data/processed/predictions.csv")

    df = derive_time_features(df, timestamp_col="attributes.arrival_time")
    detect_bias(df)