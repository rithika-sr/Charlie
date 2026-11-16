#!/usr/bin/env python3
"""
Prometheus Metrics Exporter for Charlie MBTA
Run this to expose metrics at http://localhost:8000/metrics
"""

import time
import requests
from prometheus_client import start_http_server, Gauge, Counter, Histogram

SERVICE_URL = "https://charlie-mbta-api-588293495748.us-east1.run.app"

# Define metrics
api_health = Gauge('charlie_api_health', 'API health (1=up, 0=down)')
stops_loaded = Gauge('charlie_stops_loaded', 'MBTA stops loaded')
model_available = Gauge('charlie_model_available', 'Model available (1=yes, 0=no)')
request_latency = Histogram('charlie_request_latency_seconds', 'Request latency')
requests_total = Counter('charlie_requests_total', 'Total requests')
requests_failed = Counter('charlie_requests_failed', 'Failed requests')

def collect_metrics():
    try:
        start = time.time()
        response = requests.get(f"{SERVICE_URL}/health", timeout=10)
        latency = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            api_health.set(1)
            stops_loaded.set(data.get('stops_loaded', 0))
            model_available.set(1 if data.get('ml_model_available') else 0)
            request_latency.observe(latency)
            requests_total.inc()
            print(f"✅ Metrics updated - Latency: {latency*1000:.0f}ms")
        else:
            api_health.set(0)
            requests_failed.inc()
    except Exception as e:
        api_health.set(0)
        requests_failed.inc()
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Prometheus exporter on :8000")
    print("📊 Metrics: http://localhost:8000/metrics")
    
    start_http_server(8000)
    
    while True:
        collect_metrics()
        time.sleep(30)
