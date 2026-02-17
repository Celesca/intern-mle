Sawit Koseeyaumporn

### How to Run

First, generate the data with `python 01_generate_dataset.py`

Docker way

```bash
docker-compose up --build
```

or the local way

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python server.py
# Or with multiple workers:
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4

# Run load test
python load_test.py
```

### API Endpoint

```
POST /recommend/{user_id}
```

**Request Body:**
```json
{
  "candidate_restaurant_ids": [1, 2, 3, ...],
  "latitude": 13.7563,
  "longitude": 100.5018,
  "size": 20,
  "max_dist": 5000,
  "sort_dist": false
}
```

**Response:**
```json
{
  "restaurants": [
    {"id": 2, "score": 0.85, "displacement": 800},
    {"id": 1, "score": 0.72, "displacement": 1200}
  ]
}
```

