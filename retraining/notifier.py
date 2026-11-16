#!/usr/bin/env python3
"""
Notification System for Charlie MBTA
Sends alerts when retraining is triggered or completed
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, Any

# Environment variables (set these in your environment or .env)
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"

class Notifier:
    """Send notifications via Slack, email, or console"""
    
    def __init__(self):
        self.slack_enabled = bool(SLACK_WEBHOOK_URL)
        self.email_enabled = EMAIL_ENABLED
    
    def format_message(self, event: str, details: Dict[str, Any]) -> str:
        """Format notification message"""
        
        if event == "retraining_started":
            message = f"""
🔄 CHARLIE MBTA - Retraining Started

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Reason: {details.get('reason', 'Unknown')}
Drift PSI: {details.get('max_psi', 0):.3f}
Status: Training in progress...
"""
        
        elif event == "retraining_success":
            message = f"""
✅ CHARLIE MBTA - Retraining Successful

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {details.get('duration_seconds', 0):.1f}s

New Model Performance:
  AUC: {details.get('new_auc', 0):.4f}
  Improvement: +{details.get('improvement', 0):.4f}

Status: Deployed to production via GitHub Actions
Service: https://charlie-mbta-api-588293495748.us-east1.run.app
"""
        
        elif event == "retraining_rejected":
            message = f"""
⚠️  CHARLIE MBTA - Retraining Rejected

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

New model did not improve performance.
  New AUC: {details.get('new_auc', 0):.4f}
  Old AUC: {details.get('old_auc', 0):.4f}
  Improvement: {details.get('improvement', 0):+.4f}

Status: Keeping existing model in production
"""
        
        elif event == "retraining_failed":
            message = f"""
❌ CHARLIE MBTA - Retraining Failed

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Error: {details.get('error', 'Unknown error')}

Status: Manual intervention required
Log: {details.get('log_file', 'N/A')}
"""
        
        else:
            message = f"CHARLIE MBTA Event: {event}\n{json.dumps(details, indent=2)}"
        
        return message
    
    def send_slack(self, message: str):
        """Send Slack notification"""
        if not self.slack_enabled:
            return
        
        try:
            payload = {
                "text": message,
                "username": "Charlie MBTA Bot",
                "icon_emoji": ":train:"
            }
            
            response = requests.post(
                SLACK_WEBHOOK_URL,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print("  ✅ Slack notification sent")
            else:
                print(f"  ⚠️  Slack notification failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ⚠️  Slack notification error: {e}")
    
    def send_email(self, subject: str, message: str):
        """Send email notification (placeholder)"""
        if not self.email_enabled:
            return
        
        # TODO: Implement email sending
        # Example using SendGrid, AWS SES, or Gmail SMTP
        print(f"  ℹ️  Email notification: {subject}")
        print(f"     (Email sending not configured)")
    
    def notify(self, event: str, details: Dict[str, Any]):
        """Send notification via all enabled channels"""
        message = self.format_message(event, details)
        
        # Always log to console
        print("\n" + "=" * 70)
        print("NOTIFICATION")
        print("=" * 70)
        print(message)
        
        # Send via configured channels
        self.send_slack(message)
        
        if self.email_enabled:
            subject = f"Charlie MBTA: {event.replace('_', ' ').title()}"
            self.send_email(subject, message)

# Global notifier instance
notifier = Notifier()

# Convenience functions
def notify_retraining_started(max_psi: float):
    notifier.notify("retraining_started", {"reason": "drift_detected", "max_psi": max_psi})

def notify_retraining_success(duration: float, new_auc: float, improvement: float):
    notifier.notify("retraining_success", {
        "duration_seconds": duration,
        "new_auc": new_auc,
        "improvement": improvement
    })

def notify_retraining_rejected(new_auc: float, old_auc: float, improvement: float):
    notifier.notify("retraining_rejected", {
        "new_auc": new_auc,
        "old_auc": old_auc,
        "improvement": improvement
    })

def notify_retraining_failed(error: str, log_file: str = ""):
    notifier.notify("retraining_failed", {"error": error, "log_file": log_file})

if __name__ == "__main__":
    # Test notifications
    print("Testing notification system...")
    notify_retraining_started(0.35)
