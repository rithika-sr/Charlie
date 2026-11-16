#!/bin/bash
# Runs monitoring and retraining check
cd "$(dirname "$0")/.."

echo "[$(date)] Starting scheduled monitoring..."

# Run monitoring
python monitoring/model_monitor.py >> retraining/reports/scheduled.log 2>&1

# Run drift check and retraining
python retraining/trigger_retrain.py >> retraining/reports/scheduled.log 2>&1

echo "[$(date)] Complete"
