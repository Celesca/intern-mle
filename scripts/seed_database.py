"""
Script to seed Redis database with user and restaurant features from parquet files.
Run this script to populate the database before starting the server.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from app.database import SyncFeatureDatabase
from app.config import DATA_DIR, REDIS_HOST, REDIS_PORT


def seed_database(host: str = REDIS_HOST, port: int = REDIS_PORT, batch_size: int = 1000):
    """Load all user and restaurant features into Redis."""
    
    print(f"Connecting to Redis at {host}:{port}...")
    db = SyncFeatureDatabase(host, port)
    
    # Load and store user features
    print("Loading user features from parquet...")
    user_df = pd.read_parquet(DATA_DIR / "user_features.parquet")
    user_ids = user_df['user_id'].values
    feature_cols = [c for c in user_df.columns if c != 'user_id']
    user_features = user_df[feature_cols].values.astype(np.float32)
    
    print(f"Storing {len(user_ids)} users in Redis...")
    for i in tqdm(range(0, len(user_ids), batch_size), desc="Users"):
        batch_ids = user_ids[i:i+batch_size]
        batch_features = user_features[i:i+batch_size]
        db.set_user_features_batch(batch_ids.tolist(), batch_features)
    
    del user_df, user_features
    
    # Load and store restaurant features
    print("Loading restaurant features from parquet...")
    restaurant_df = pd.read_parquet(DATA_DIR / "restaurant_features.parquet")
    restaurant_ids = restaurant_df['restaurant_id'].values
    latitudes = restaurant_df['latitude'].values.astype(np.float64)
    longitudes = restaurant_df['longitude'].values.astype(np.float64)
    feature_cols = [c for c in restaurant_df.columns if c not in ['restaurant_id', 'latitude', 'longitude']]
    restaurant_features = restaurant_df[feature_cols].values.astype(np.float32)
    
    print(f"Storing {len(restaurant_ids)} restaurants in Redis...")
    for i in tqdm(range(0, len(restaurant_ids), batch_size), desc="Restaurants"):
        batch_ids = restaurant_ids[i:i+batch_size]
        batch_features = restaurant_features[i:i+batch_size]
        batch_lats = latitudes[i:i+batch_size]
        batch_lons = longitudes[i:i+batch_size]
        db.set_restaurant_features_batch(batch_ids.tolist(), batch_features, batch_lats, batch_lons)
    
    db.close()
    print("Database seeding complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed Redis database with features")
    parser.add_argument("--host", default=REDIS_HOST, help="Redis host")
    parser.add_argument("--port", type=int, default=REDIS_PORT, help="Redis port")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for inserts")
    args = parser.parse_args()
    
    seed_database(args.host, args.port, args.batch_size)
