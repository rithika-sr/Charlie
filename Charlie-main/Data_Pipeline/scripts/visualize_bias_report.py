# File: Data_Pipeline/scripts/visualize_bias_report.py

import pandas as pd
import plotly.graph_objects as go
import logging
from pathlib import Path

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
# Visualization function
# -------------------------------------------------------
def visualize_bias_report(input_csv, output_html):
    setup_logger()
    logging.info(f"📊 Loading bias report → {input_csv}")

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error("❌ bias_report.csv not found. Run detect_bias.py first.")
        return
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return

    if "feature" not in df.columns:
        logging.error("❌ Missing required column 'feature' in bias_report.csv.")
        return

    # Auto-detect grouping column
    group_col = "feature" if "feature" in df.columns else df.columns[0]
    metric_cols = [
        c for c in df.columns if c not in ["feature", "Unnamed: 0", group_col]
    ]

    # Convert metrics to numeric (ignore errors for non-numeric values)
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop completely empty metric columns
    metric_cols = [c for c in metric_cols if df[c].notna().any()]

    if not metric_cols:
        logging.error("❌ No valid numeric metrics to visualize.")
        return

    fig = go.Figure()

    for metric in metric_cols:
        fig.add_trace(
            go.Bar(
                x=df[group_col],
                y=df[metric],
                name=metric.capitalize(),
                text=df[metric].round(3).astype(str),
                textposition="auto",
            )
        )

    fig.update_layout(
        title="Bias Metrics by Route / Feature",
        xaxis_title="Feature / Group",
        yaxis_title="Metric Value",
        barmode="group",
        template="plotly_white",
        legend_title="Metrics",
        width=1000,
        height=600,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html))
    logging.info(f"✅ Bias visualization saved → {output_html}")


# -------------------------------------------------------
# Main execution
# -------------------------------------------------------
if __name__ == "__main__":
    input_csv = Path("Data_Pipeline/data/validation/bias_report.csv")
    output_html = Path("Data_Pipeline/data/validation/bias_visualization.html")
    visualize_bias_report(input_csv, output_html)