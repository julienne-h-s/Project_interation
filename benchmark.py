import requests
import time
import json
import os

TEST_DIR = "test_audio"  
API_URL = "http://localhost:5000/predict"

results = []

for fname in os.listdir(TEST_DIR):
    if not fname.endswith(".wav"):
        continue

    path = os.path.join(TEST_DIR, fname)
    files = {"file": open(path, "rb")}

    start = time.time()
    r = requests.post(API_URL, files=files)
    end = time.time()

    latency = (end - start) * 1000

    if r.status_code == 200:
        pred = r.json()["command"]
    else:
        pred = "ERROR"

    results.append({
        "file": fname,
        "prediction": pred,
        "latency_ms": latency
    })

metrics = {
    "avg_latency_ms": sum(r["latency_ms"] for r in results) / len(results),
    "results": results
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Benchmark complete! Saved to metrics.json")
