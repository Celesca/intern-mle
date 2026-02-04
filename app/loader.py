import numpy as np
import pandas as pd
import torch

from config import DATA_DIR


def load_model() -> torch.jit.ScriptModule:
    model = torch.jit.load(DATA_DIR / "model.pt")
    model.eval()
    return model


def load_user_features() -> tuple[np.ndarray, dict]:
    user_df = pd.read_parquet(DATA_DIR / "user_features.parquet")
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_df['user_id'].values)}
    feature_cols = [c for c in user_df.columns if c != 'user_id']
    user_features = user_df[feature_cols].values.astype(np.float32)
    return user_features, user_id_to_idx


def load_restaurant_features() -> tuple[np.ndarray, np.ndarray, dict]:
    restaurant_df = pd.read_parquet(DATA_DIR / "restaurant_features.parquet")
    restaurant_id_to_idx = {rid: idx for idx, rid in enumerate(restaurant_df['restaurant_id'].values)}
    restaurant_locations = restaurant_df[['latitude', 'longitude']].values.astype(np.float64)
    feature_cols = [c for c in restaurant_df.columns if c not in ['restaurant_id', 'latitude', 'longitude']]
    restaurant_features = restaurant_df[feature_cols].values.astype(np.float32)
    return restaurant_features, restaurant_locations, restaurant_id_to_idx
