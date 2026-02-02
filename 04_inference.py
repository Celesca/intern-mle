import json
from pathlib import Path
import numpy as np
import torch
import pandas as pd


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


DATA_DIR = Path("data")

request_df = pd.read_parquet(DATA_DIR / "requests.parquet")
user_df = pd.read_parquet(DATA_DIR / "user_features.parquet")
restaurant_df = pd.read_parquet(DATA_DIR / "restaurant_features.parquet").set_index(
    "restaurant_id", drop=True
)

model = torch.jit.load(DATA_DIR / "model.pt")
model.eval()

request = request_df.iloc[0]
user_features = (
    user_df[user_df["user_id"] == request["user_id"]].drop(columns="user_id").values
)
restaurant_features = (
    restaurant_df.loc[request["candidate_restaurant_ids"]]
    .drop(columns=["latitude", "longitude"])
    .values
)
x = torch.tensor(
    np.hstack(
        (
            np.tile(user_features, (len(request["candidate_restaurant_ids"]), 1)),
            restaurant_features,
        )
    ),
    dtype=torch.float32,
)

with torch.no_grad():
    y_pred = torch.sigmoid(model(x))

result = [
    {"restaurant_id": rid, "score": prob}
    for rid, prob in zip(
        request["candidate_restaurant_ids"],
        y_pred.numpy(),
    )
]
sorted_result = sorted(result, key=lambda item: item["score"], reverse=True)
print(
    json.dumps(
        sorted_result,
        indent=4,
        cls=NpEncoder,
    )
)
