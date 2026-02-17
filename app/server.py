import math
from typing import List
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import redis.asyncio as redis

from .config import REDIS_HOST, REDIS_PORT, EARTH_RADIUS_METERS
from .loader import load_model
from .database import FeatureDatabase
from .h3_utils import filter_by_h3_proximity


class RecommendRequest(BaseModel):
    candidate_restaurant_ids: List[int]
    latitude: float
    longitude: float
    size: int = Field(default=20, ge=1)
    max_dist: float = Field(default=5000, ge=0)  # meters
    sort_dist: bool = Field(default=False)


class Restaurant(BaseModel):
    id: int
    score: float
    displacement: float


class RecommendResponse(BaseModel):
    restaurants: List[Restaurant]


class AppState:
    model: torch.jit.ScriptModule = None
    redis_client: redis.Redis = None
    db: FeatureDatabase = None


state = AppState()

# calculate haversine distance - use numpy vectorization for speed
def haversine_distance(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:


    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return EARTH_RADIUS_METERS * c


async def run_inference(
    user_id: str,
    request: RecommendRequest
) -> RecommendResponse:
    
    # Parse user_id (e.g., "u00000" -> 0)
    try:
        user_id_int = int(user_id.lstrip('u'))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid user_id format: {user_id}")
    
    # Get user features from database
    user_features = await state.db.get_user_features(user_id_int)
    if user_features is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    # Get restaurant data from database (batch query)
    rest_features, rest_lats, rest_lons, valid_ids = await state.db.get_restaurants_batch(
        request.candidate_restaurant_ids
    )
    
    if not valid_ids:
        return RecommendResponse(restaurants=[])
    
    # Step 1: H3 pre-filter for fast proximity check
    # This quickly eliminates restaurants outside the search radius using hexagonal grid
    h3_filtered_indices = filter_by_h3_proximity(
        request.latitude, request.longitude,
        rest_lats.tolist(), rest_lons.tolist(),
        valid_ids,
        request.max_dist
    )
    
    if not h3_filtered_indices:
        return RecommendResponse(restaurants=[])
    
    # Apply H3 pre-filter
    h3_filtered_ids = [valid_ids[i] for i in h3_filtered_indices]
    h3_filtered_features = rest_features[h3_filtered_indices]
    h3_filtered_lats = rest_lats[h3_filtered_indices]
    h3_filtered_lons = rest_lons[h3_filtered_indices]
    
    # Step 2: Calculate exact haversine distances (only for H3-filtered candidates)
    distances = haversine_distance(
        request.latitude, request.longitude,
        h3_filtered_lats, h3_filtered_lons
    )
    
    # Step 3: Final filter by exact max_dist
    within_range_mask = distances <= request.max_dist
    
    if not np.any(within_range_mask):
        return RecommendResponse(restaurants=[])
    
    # Apply exact distance filter
    filtered_indices = np.where(within_range_mask)[0]
    filtered_ids = [h3_filtered_ids[i] for i in filtered_indices]
    filtered_features = h3_filtered_features[filtered_indices]
    filtered_distances = distances[filtered_indices]
    
    # Prepare input tensor for model
    # Tile user features to match number of restaurants
    num_restaurants = len(filtered_ids)
    user_tiled = np.tile(user_features, (num_restaurants, 1))
    
    # Concatenate user and restaurant features
    x = np.hstack([user_tiled, filtered_features]).astype(np.float32)
    x_tensor = torch.from_numpy(x)
    
    # Run inference
    with torch.no_grad():
        logits = state.model(x_tensor)
        scores = torch.sigmoid(logits).numpy()
    
    # Create result list
    results = []
    for i, (rid, score, dist) in enumerate(zip(filtered_ids, scores, filtered_distances)):
        results.append(Restaurant(
            id=rid,
            score=float(score),
            displacement=float(dist)
        ))
    
    # Sort results
    if request.sort_dist:
        results.sort(key=lambda r: r.displacement)
    else:
        results.sort(key=lambda r: r.score, reverse=True)
    
    # Limit to requested size
    results = results[:request.size]
    
    return RecommendResponse(restaurants=results)

@asynccontextmanager
async def lifespan(app: FastAPI):
    state.model = load_model()
    
    # db
    try:
        state.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=False
        )
        await state.redis_client.ping()
        state.db = FeatureDatabase(state.redis_client)
        print("Redis database connected successfully")
    except Exception as e:
        print(f"Redis connection failed: {e}")
        raise RuntimeError("Cannot start server without Redis database")
    
    yield
    
    # Cleanup
    print("Shutdown server")
    if state.redis_client:
        await state.redis_client.close()

app = FastAPI(
    title="Restaurant Recommendation API",
    description="HTTP API for restaurant recommendations using ML model inference",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend(user_id: str, request: RecommendRequest):
    return await run_inference(user_id, request)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "redis": state.redis_client is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        workers=2,  
        reload=False
    )
