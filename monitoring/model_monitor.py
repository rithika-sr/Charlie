import requests
import json
from datetime import datetime
from pathlib import Path

SERVICE_URL = "https://charlie-mbta-api-588293495748.us-east1.run.app"

def check_health():
    response = requests.get(f"{SERVICE_URL}/health")
    data = response.json()
    print(f"✅ Health: {data['status']}")
    print(f"✅ Stops: {data['stops_loaded']:,}")
    print(f"✅ Model: {data['ml_model_available']}")
    return data

def test_latency(num=10):
    import time
    latencies = []
    for _ in range(num):
        start = time.time()
        requests.get(f"{SERVICE_URL}/health")
        latencies.append((time.time() - start) * 1000)
    
    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95)]
    print(f"\n⏱️  Latency P95: {p95:.0f}ms")
    return p95

print("=" * 60)
print("CHARLIE MBTA - MONITORING REPORT")
print("=" * 60)
print(f"Time: {datetime.now()}\n")

health = check_health()
latency = test_latency()

print("\n✅ Monitoring complete!")