#!/usr/bin/env python3
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
drift_report = project_root / "Model_Development" / "reports" / "drift_report.json"

print("\n🔍 Checking for data drift...")

if drift_report.exists():
    with open(drift_report) as f:
        data = json.load(f)
    
    psi_scores = data.get('psi_scores', {})
    max_psi = max(psi_scores.values()) if psi_scores else 0
    
    print(f"  Max PSI: {max_psi:.3f}")
    
    if max_psi < 0.1:
        print(f"  ✅ No drift detected")
    elif max_psi < 0.3:
        print(f"  ⚠️  Minor drift detected")
    else:
        print(f"  ❌ Major drift - Retraining recommended!")
else:
    print(f"  ℹ️  No drift report found")
