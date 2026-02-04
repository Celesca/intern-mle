import math
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import redis.asyncio as redis
import pickle

from config import REDIS_HOST, REDIS_PORT, CACHE_TTL, EARTH_RADIUS_METERS
from loader import load_model, load_user_features, load_restaurant_features


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
    user_features: np.ndarray = None
    user_id_to_idx: dict = None
    restaurant_features: np.ndarray = None
    restaurant_locations: np.ndarray = None  # (lat, lon) pairs
    restaurant_id_to_idx: dict = None


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


def get_cache_key(user_id: str, request: RecommendRequest) -> str:

    candidates_hash = hash(tuple(sorted(request.candidate_restaurant_ids)))
    return f"""recommend:{user_id}
                        :{candidates_hash}
                        :{request.latitude:.6f}
                        :{request.longitude:.6f}
                        :{request.size}
                        :{request.max_dist}
                        :{request.sort_dist}"""

async def get_user_features(user_id: int) -> Optional[np.ndarray]:

    cache_key = f"user:{user_id}"
    
    if state.redis_client:
        try:
            cached = await state.redis_client.get(cache_key)
            if cached:
                return pickle.loads(cached)
        except Exception:
            pass 
    
    if user_id in state.user_id_to_idx: # Get from "database" (pre-loaded numpy array)
        idx = state.user_id_to_idx[user_id]
        features = state.user_features[idx]
        
        if state.redis_client: # Cache
            try:
                await state.redis_client.setex(cache_key, CACHE_TTL, pickle.dumps(features))
            except Exception:
                pass
        
        return features
    
    return None


async def get_restaurant_data(restaurant_ids: List[int]) -> tuple:

    valid_ids = []
    valid_indices = []
    
    for rid in restaurant_ids:
        if rid in state.restaurant_id_to_idx:
            valid_ids.append(rid)
            valid_indices.append(state.restaurant_id_to_idx[rid])
    
    if not valid_ids:
        return None, None, None, []
    
    valid_indices = np.array(valid_indices)
    
    features = state.restaurant_features[valid_indices]

    locations = state.restaurant_locations[valid_indices]
    latitudes = locations[:, 0]
    longitudes = locations[:, 1]
    
    return features, latitudes, longitudes, valid_ids

async def run_inference(
    user_id: str,
    request: RecommendRequest
) -> RecommendResponse:

    # Parse user_id (e.g., "u00000" -> 0)
    try:
        user_id_int = int(user_id.lstrip('u'))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid user_id format: {user_id}")
    
    # Get user features
    user_features = await get_user_features(user_id_int)
    if user_features is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    # Get restaurant data
    rest_features, rest_lats, rest_lons, valid_ids = await get_restaurant_data(
        request.candidate_restaurant_ids
    )
    
    if not valid_ids:
        return RecommendResponse(restaurants=[])
    
    # Calculate distances (vectorized)
    distances = haversine_distance(
        request.latitude, request.longitude,
        rest_lats, rest_lons
    )
    
    # Filter by max_dist
    within_range_mask = distances <= request.max_dist
    
    if not np.any(within_range_mask):
        return RecommendResponse(restaurants=[])
    
    # Apply filter
    filtered_indices = np.where(within_range_mask)[0]
    filtered_ids = [valid_ids[i] for i in filtered_indices]
    filtered_features = rest_features[filtered_indices]
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
    
    state.user_features, state.user_id_to_idx = load_user_features()
    state.restaurant_features, state.restaurant_locations, state.restaurant_id_to_idx = load_restaurant_features()
    
    # Redis caching
    try:
        state.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=False
        )
        await state.redis_client.ping()
        print("Redis connected successfully")
    except Exception as e:
        print(f"Redis connection failed: {e}")
        print("Continuing without Redis caching...")
        state.redis_client = None
    
    yield
    
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
