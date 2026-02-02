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
BATCH_SIZE = 1024
EPOCHS = 5
INPUT_DIM = 30 + 10  # user + restaurant features


# DO NOT EDIT MODEL ARCHITECTURE
class RankNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)


def train():
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.parquet")))
    user_features = pd.read_parquet(DATA_DIR / "user_features.parquet")
    restaurant_features = pd.read_parquet(DATA_DIR / "restaurant_features.parquet")
    train_df = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)
    train_df = train_df.merge(user_features, on=["user_id"]).merge(
        restaurant_features, on=["restaurant_id"]
    )
    num_batches = math.ceil(len(train_df) / BATCH_SIZE)

    model = RankNet(INPUT_DIM)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-8)

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        for batch in tqdm(range(num_batches), desc=f"epoch {epoch + 1}"):
            batch_df = train_df.iloc[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
            x = torch.tensor(
                batch_df.drop(
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
            y = torch.tensor(batch_df["click"].values.astype(np.float32))
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        print(f"Loss: {loss.item():.4f}, Elapsed time {time.time() - start_time:.3f} seconds")
    model_scripted = torch.jit.script(model)
    model_scripted.save(DATA_DIR / "model.pt")


if __name__ == "__main__":
    train()
