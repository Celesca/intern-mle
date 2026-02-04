# load_test.py - Load testing script for the recommendation API
import asyncio
import time
import statistics
from pathlib import Path
import pandas as pd
import httpx
import json

DATA_DIR = Path("data")
BASE_URL = "http://localhost:8000"
TARGET_RPS = 30  # Requests per second
DURATION_SECONDS = 60  # Test duration
TARGET_P95_MS = 100  # Target 95th percentile in milliseconds


async def make_request(client: httpx.AsyncClient, request_data: dict) -> tuple:
    """Make a single request and return (success, latency_ms)."""
    user_id = f"u{request_data['user_id']:05d}"
    url = f"{BASE_URL}/recommend/{user_id}"
    
    # Convert numpy arrays to lists for JSON serialization
    candidate_ids = request_data["candidate_restaurant_ids"]
    if hasattr(candidate_ids, 'tolist'):
        candidate_ids = candidate_ids.tolist()
    elif not isinstance(candidate_ids, list):
        candidate_ids = list(candidate_ids)
    
    payload = {
        "candidate_restaurant_ids": candidate_ids,
        "latitude": float(request_data["latitude"]),
        "longitude": float(request_data["longitude"]),
        "size": int(request_data["size"]),
        "max_dist": float(request_data["max_dist"]),
        "sort_dist": bool(request_data["sort_dist"])
    }
    
    start_time = time.perf_counter()
    try:
        response = await client.post(url, json=payload, timeout=10.0)
        latency_ms = (time.perf_counter() - start_time) * 1000
        success = response.status_code == 200
        return success, latency_ms
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        print(f"Request error: {e}")
        return False, latency_ms


async def run_load_test():
    """Run the load test."""
    print("=" * 60)
    print("LOAD TEST - Restaurant Recommendation API")
    print("=" * 60)
    print(f"Target: {TARGET_RPS} requests/second for {DURATION_SECONDS} seconds")
    print(f"Target P95 latency: {TARGET_P95_MS}ms")
    print()
    
    # Load request data
    print("Loading test data...")
    requests_df = pd.read_parquet(DATA_DIR / "requests.parquet")
    requests_list = requests_df.to_dict('records')
    num_requests = len(requests_list)
    print(f"Loaded {num_requests} test requests")
    print()
    
    # Calculate timing
    total_requests = TARGET_RPS * DURATION_SECONDS
    interval = 1.0 / TARGET_RPS
    
    print(f"Starting load test...")
    print(f"Total requests to send: {total_requests}")
    print()
    
    latencies = []
    successes = 0
    failures = 0
    
    async with httpx.AsyncClient() as client:
        # Warm up
        print("Warming up (5 requests)...")
        for i in range(5):
            req = requests_list[i % num_requests]
            await make_request(client, req)
        print()
        
        # Main test
        start_time = time.perf_counter()
        tasks = []
        request_count = 0
        
        print("Running load test...")
        
        while request_count < total_requests:
            # Calculate when this request should be sent
            expected_time = start_time + (request_count * interval)
            current_time = time.perf_counter()
            
            # Wait if we're ahead of schedule
            if current_time < expected_time:
                await asyncio.sleep(expected_time - current_time)
            
            # Get request data (cycle through available requests)
            req = requests_list[request_count % num_requests]
            
            # Send request
            task = asyncio.create_task(make_request(client, req))
            tasks.append(task)
            request_count += 1
            
            # Print progress every 100 requests
            if request_count % 100 == 0:
                elapsed = time.perf_counter() - start_time
                actual_rps = request_count / elapsed if elapsed > 0 else 0
                print(f"  Sent {request_count}/{total_requests} requests, actual RPS: {actual_rps:.1f}")
        
        # Wait for all requests to complete
        print("\nWaiting for all requests to complete...")
        results = await asyncio.gather(*tasks)
        
        total_time = time.perf_counter() - start_time
    
    # Process results
    for success, latency in results:
        latencies.append(latency)
        if success:
            successes += 1
        else:
            failures += 1
    
    # Calculate statistics
    latencies.sort()
    avg_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p50_latency = latencies[int(len(latencies) * 0.50)]
    p90_latency = latencies[int(len(latencies) * 0.90)]
    p95_latency = latencies[int(len(latencies) * 0.95)]
    p99_latency = latencies[int(len(latencies) * 0.99)]
    actual_rps = len(latencies) / total_time
    
    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total requests:     {len(latencies)}")
    print(f"Successful:         {successes}")
    print(f"Failed:             {failures}")
    print(f"Success rate:       {successes/len(latencies)*100:.2f}%")
    print()
    print(f"Total time:         {total_time:.2f}s")
    print(f"Actual RPS:         {actual_rps:.2f}")
    print()
    print("Latency (ms):")
    print(f"  Min:              {min_latency:.2f}")
    print(f"  Avg:              {avg_latency:.2f}")
    print(f"  P50:              {p50_latency:.2f}")
    print(f"  P90:              {p90_latency:.2f}")
    print(f"  P95:              {p95_latency:.2f}")
    print(f"  P99:              {p99_latency:.2f}")
    print(f"  Max:              {max_latency:.2f}")
    print()
    
    # Check if requirements are met
    print("=" * 60)
    print("REQUIREMENT CHECK")
    print("=" * 60)
    
    rps_pass = actual_rps >= TARGET_RPS * 0.95  # Allow 5% tolerance
    p95_pass = p95_latency <= TARGET_P95_MS
    
    print(f"RPS >= {TARGET_RPS}:          {'✓ PASS' if rps_pass else '✗ FAIL'} ({actual_rps:.2f})")
    print(f"P95 <= {TARGET_P95_MS}ms:        {'✓ PASS' if p95_pass else '✗ FAIL'} ({p95_latency:.2f}ms)")
    print()
    
    if rps_pass and p95_pass:
        print("🎉 ALL REQUIREMENTS MET!")
    else:
        print("⚠️  Some requirements not met. Consider:")
        if not rps_pass:
            print("   - Increase uvicorn workers")
            print("   - Use gunicorn with multiple workers")
        if not p95_pass:
            print("   - Enable Redis caching")
            print("   - Optimize model inference (batch processing)")
            print("   - Use GPU acceleration if available")
    
    return rps_pass and p95_pass


if __name__ == "__main__":
    asyncio.run(run_load_test())
