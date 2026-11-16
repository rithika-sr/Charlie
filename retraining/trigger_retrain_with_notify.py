#!/usr/bin/env python3
"""Automated Retraining with Slack Notifications"""

import sys
from pathlib import Path
from trigger_retrain import RetrainingPipeline
from notifier import notify_retraining_started, notify_retraining_success, notify_retraining_rejected, notify_retraining_failed

class RetrainingWithNotifications(RetrainingPipeline):
    def run(self):
        try:
            drift_detected = self.check_drift()
            
            if not drift_detected:
                self.log("\n✅ No retraining needed")
                return 0
            
            # Notify that retraining started
            notify_retraining_started(0.35)  # Get actual PSI from drift check
            
            self.log("\n🔄 RETRAINING TRIGGERED")
            self.run_training()
            
            if self.validate_model():
                self.deploy()
                notify_retraining_success(60.0, 0.92, 0.02)  # Get from validation
                self.log("\n✅ COMPLETE - Notification sent to Slack")
                return 0
            else:
                notify_retraining_rejected(0.90, 0.91, -0.01)
                self.log("\n⚠️  Keeping existing - Notification sent to Slack")
                return 1
                
        except Exception as e:
            notify_retraining_failed(str(e), str(self.log_file))
            raise

if __name__ == "__main__":
    sys.exit(RetrainingWithNotifications().run())
