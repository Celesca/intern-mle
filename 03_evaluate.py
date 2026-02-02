# train_model.py
import gc
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm

DATA_DIR = Path("data")

def evaluate():
    test_files = sorted(glob.glob(str(DATA_DIR / "test" / "*.parquet")))
    user_features = pd.read_parquet(DATA_DIR / "user_features.parquet")
    restaurant_features = pd.read_parquet(DATA_DIR / "restaurant_features.parquet")

    model = torch.jit.load(DATA_DIR / "model.pt")

    test_df = pd.concat([pd.read_parquet(f) for f in test_files], ignore_index=True)
    test_df = test_df.merge(user_features, on=["user_id"]).merge(
        restaurant_features, on=["restaurant_id"]
    )
    model.eval()
    with torch.no_grad():
        x = torch.tensor(
            test_df.drop(
                columns=[
                    "user_id",
                    "restaurant_id",
                    "click",
                    "latitude",
                    "longitude",
                ]
            ).values,
            dtype=torch.float32,
        )
        y = torch.tensor(test_df["click"].values.astype(np.float32))
        output = torch.sigmoid(model(x))
        pred = (output > 0.5).float()
        correct = (pred == y).sum().item()
        total = y.size(0)
    print(f"Test Accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    evaluate()
