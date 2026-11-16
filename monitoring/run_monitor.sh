#!/bin/bash
cd "$(dirname "$0")"
python model_monitor.py >> reports/monitor.log 2>&1
echo "$(date): Monitoring complete" >> reports/monitor.log
