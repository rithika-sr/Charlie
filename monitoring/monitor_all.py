#!/usr/bin/env python3
"""Combined Monitoring Dashboard"""
import subprocess
import sys

print("=" * 70)
print("CHARLIE MBTA - COMPREHENSIVE MONITORING DASHBOARD")
print("=" * 70)

# Run service monitor
print("\n[1/2] SERVICE HEALTH & PERFORMANCE")
print("-" * 70)
subprocess.run([sys.executable, "model_monitor.py"])

# Run drift detection
print("\n[2/2] DATA DRIFT DETECTION")
print("-" * 70)
subprocess.run([sys.executable, "drift_monitor.py"])

print("\n" + "=" * 70)
print("✅ MONITORING COMPLETE")
print("=" * 70)
