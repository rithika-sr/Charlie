#!/usr/bin/env python3
"""Automated Retraining Pipeline"""

import os
import sys
import json
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DEV_DIR = PROJECT_ROOT / "Model_Development"
API_SERVICE_DIR = PROJECT_ROOT / "API_Service"

DRIFT_THRESHOLD = 0.3

class RetrainingPipeline:
    def __init__(self):
        log_dir = PROJECT_ROOT / "retraining" / "reports"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def log(self, msg):
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(f"{msg}\n")
    
    def check_drift(self):
        self.log("\n[1/5] Checking drift...")
        drift_report = MODEL_DEV_DIR / "reports" / "drift_report.json"
        
        if drift_report.exists():
            with open(drift_report) as f:
                data = json.load(f)
            psi = max(data.get('psi_scores', {}).values(), default=0)
            
            if psi > DRIFT_THRESHOLD:
                self.log(f"  ❌ Drift detected: PSI={psi:.3f}")
                return True
            else:
                self.log(f"  ✅ No drift: PSI={psi:.3f}")
        return False
    
    def run_training(self):
        self.log("\n[2/5] Training model...")
        scripts = ["model_train.py", "model_tuning.py", "model_select.py"]
        
        for script in scripts:
            path = MODEL_DEV_DIR / "ml_src" / script
            if path.exists():
                self.log(f"  Running {script}...")
                subprocess.run([sys.executable, str(path)], cwd=path.parent)
    
    def validate_model(self):
        self.log("\n[3/5] Validating...")
        comparison = MODEL_DEV_DIR / "reports" / "model_comparison.json"
        
        if comparison.exists():
            with open(comparison) as f:
                data = json.load(f)
            new_auc = data.get("baseline_lgbm_auc", 0)
            old_auc = data.get("final_model_existing_auc", 0)
            
            self.log(f"  New: {new_auc:.4f}, Old: {old_auc:.4f}")
            return new_auc > old_auc
        return False
    
    def deploy(self):
        self.log("\n[4/5] Deploying...")
        src = MODEL_DEV_DIR / "models" / "final_model.joblib"
        dst = API_SERVICE_DIR / "models" / "final_model.joblib"
        
        shutil.copy(src, dst)
        self.log(f"  ✅ Model deployed")
        
        try:
            subprocess.run(["git", "add", str(dst)], cwd=PROJECT_ROOT)
            subprocess.run(["git", "commit", "-m", "Auto-retrain"], cwd=PROJECT_ROOT)
            subprocess.run(["git", "push"], cwd=PROJECT_ROOT)
            self.log(f"  ✅ Pushed to GitHub (auto-deploy via CI/CD)")
        except:
            self.log(f"  ⚠️  Manual deployment needed")
    
    def run(self):
        self.log("=" * 60)
        self.log("AUTOMATED RETRAINING PIPELINE")
        self.log("=" * 60)
        
        if not self.check_drift():
            self.log("\n✅ No retraining needed")
            return 0
        
        self.log("\n🔄 RETRAINING TRIGGERED")
        self.run_training()
        
        if self.validate_model():
            self.deploy()
            self.log("\n✅ COMPLETE")
            return 0
        else:
            self.log("\n⚠️  Keeping existing model")
            return 1

if __name__ == "__main__":
    sys.exit(RetrainingPipeline().run())
