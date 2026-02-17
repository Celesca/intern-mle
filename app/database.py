import pickle
from typing import Optional, List, Tuple
import numpy as np
import redis.asyncio as aioredis
import redis as sync_redis


class FeatureDatabase:
    """Async Redis-based feature database for user and restaurant data."""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
    
    async def get_user_features(self, user_id: int) -> Optional[np.ndarray]:
        key = f"user:{user_id}"
        data = await self.redis.get(key)
        if data:
            return pickle.loads(data)
        return None
    
    async def get_restaurant_features(self, restaurant_id: int) -> Optional[dict]:
        key = f"restaurant:{restaurant_id}"
        data = await self.redis.get(key)
        if data:
            return pickle.loads(data)
        return None
    
    async def get_restaurants_batch(self, restaurant_ids: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        if not restaurant_ids:
            return None, None, None, []
        
        # Use pipeline for batch retrieval
        pipe = self.redis.pipeline()
        for rid in restaurant_ids:
            pipe.get(f"restaurant:{rid}")
        results = await pipe.execute()
        
        valid_ids = []
        features_list = []
        lats = []
        lons = []
        
        for rid, data in zip(restaurant_ids, results):
            if data:
                record = pickle.loads(data)
                valid_ids.append(rid)
                features_list.append(record['features'])
                lats.append(record['latitude'])
                lons.append(record['longitude'])
        
        if not valid_ids:
            return None, None, None, []
        
        return (
            np.array(features_list, dtype=np.float32),
            np.array(lats, dtype=np.float64),
            np.array(lons, dtype=np.float64),
            valid_ids
        )


class SyncFeatureDatabase:
    
    def __init__(self, host: str, port: int):
        self.redis = sync_redis.Redis(host=host, port=port, decode_responses=False)
    
    def set_user_features(self, user_id: int, features: np.ndarray):
        key = f"user:{user_id}"
        self.redis.set(key, pickle.dumps(features))
    
    def set_restaurant_features(self, restaurant_id: int, features: np.ndarray, latitude: float, longitude: float):
        key = f"restaurant:{restaurant_id}"
        data = {
            'features': features,
            'latitude': latitude,
            'longitude': longitude
        }
        self.redis.set(key, pickle.dumps(data))
    
    def set_user_features_batch(self, user_ids: List[int], features: np.ndarray):
        pipe = self.redis.pipeline()
        for i, uid in enumerate(user_ids):
            pipe.set(f"user:{uid}", pickle.dumps(features[i]))
        pipe.execute()
    
    def set_restaurant_features_batch(self, restaurant_ids: List[int], features: np.ndarray, 
                                       latitudes: np.ndarray, longitudes: np.ndarray):
        pipe = self.redis.pipeline()
        for i, rid in enumerate(restaurant_ids):
            data = {
                'features': features[i],
                'latitude': float(latitudes[i]),
                'longitude': float(longitudes[i])
            }
            pipe.set(f"restaurant:{rid}", pickle.dumps(data))
        pipe.execute()
    
    def close(self):
        self.redis.close()
